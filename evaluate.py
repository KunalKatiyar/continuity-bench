#!/usr/bin/env python3
"""Evaluation harness for the continuity benchmark.

Subcommands
    run     score a predictor over a built corpus and write a results JSON
    list    show the registered predictors
    attack  run every zero-cost heuristic predictor, to show what the corpus leaks
            (heuristics only - never a model, whose cost is money or wall clock)

A predictor is a callable taking one corpus item and returning a Prediction. The
point of the cheap heuristic predictors is adversarial: if a regex scoring no LLM
calls gets a high F1, the corpus is measuring an artefact of injection rather than
continuity reasoning, and needs fixing before anyone pays for a real eval.

ponytail: metrics are counted by hand rather than pulled from sklearn, because the
whole set is four counters and a division, and this file stays dependency-free.
"""

from __future__ import annotations

import argparse
import json
import re
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import bench

ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "corpus" / "continuity_v0.jsonl"
RESULTS_DIR = ROOT / "results"


@dataclass
class Prediction:
    """One predictor's verdict on one passage."""

    has_error: bool
    paragraph_index: int | None = None
    cost_usd: float = 0.0
    confidence: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    note: str = ""


INCONCLUSIVE_WIDTH = 0.20


def j_verdict(low, high):
    """Classify a J interval: an effect, a measured null, or not enough data.

    A tight interval around zero and a wide one around zero mean opposite things - the
    first establishes that a model does not discriminate, the second establishes
    nothing at all - and collapsing them into "not significant" hides the difference.
    """
    if low > 0 or high < 0:
        return "effect"
    return "inconclusive" if (high - low) > INCONCLUSIVE_WIDTH else "no_effect"


def wilson_interval(successes, trials, z=1.96):
    """95% Wilson score interval for a proportion, or (0, 0) with no trials.

    Plain +/- sqrt(p(1-p)/n) misbehaves near 0 and 1 and at small n, which is exactly
    where a benchmark run sits: 20 injected items is enough to read a headline number
    off and wrong enough to mislead.
    """
    if trials == 0:
        return 0.0, 0.0
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


@dataclass
class Scores:
    """Counted outcomes plus the derived metrics the leaderboard reports."""

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    localized: int = 0
    cost_usd: float = 0.0
    confidence: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latencies: list = field(default_factory=list)
    per_rule: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    per_novel: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0]))

    @property
    def precision(self):
        flagged = self.true_positive + self.false_positive
        return self.true_positive / flagged if flagged else 0.0

    @property
    def recall(self):
        actual = self.true_positive + self.false_negative
        return self.true_positive / actual if actual else 0.0

    @property
    def f1(self):
        if not self.precision or not self.recall:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def false_positive_rate(self):
        clean = self.false_positive + self.true_negative
        return self.false_positive / clean if clean else 0.0

    @property
    def youden_j(self):
        """Recall minus false-positive rate: discrimination, immune to flagging everything.

        F1 is the wrong leakage test on a paired corpus, where flagging every passage
        scores precision 0.5, recall 1.0 and so F1 0.667 while discriminating nothing.
        J is 0 for that strategy and 1 for a perfect one.
        """
        return self.recall - self.false_positive_rate

    @property
    def localization(self):
        return self.localized / self.true_positive if self.true_positive else 0.0

    @property
    def recall_interval(self):
        return wilson_interval(self.true_positive, self.true_positive + self.false_negative)

    @property
    def fpr_interval(self):
        return wilson_interval(self.false_positive, self.false_positive + self.true_negative)

    @property
    def youden_j_interval(self):
        """Worst and best case J, combining the recall and FPR intervals.

        A J interval spanning zero means the run cannot tell "discriminates nothing"
        from "discriminates a little", however tidy the point estimate looks.
        """
        recall_lo, recall_hi = self.recall_interval
        fpr_lo, fpr_hi = self.fpr_interval
        return recall_lo - fpr_hi, recall_hi - fpr_lo

    def summary(self):
        j_lo, j_hi = self.youden_j_interval
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "youden_j": round(self.youden_j, 4),
            "youden_j_ci95": [round(j_lo, 4), round(j_hi, 4)],
            "youden_j_significant": bool(j_lo > 0 or j_hi < 0),
            "youden_j_verdict": j_verdict(j_lo, j_hi),
            "recall_ci95": [round(v, 4) for v in self.recall_interval],
            "fpr_ci95": [round(v, 4) for v in self.fpr_interval],
            "false_positive_rate": round(self.false_positive_rate, 4),
            "localization": round(self.localization, 4),
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
            "cost_usd": round(self.cost_usd, 6),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "median_latency_s": round(statistics.median(self.latencies), 4) if self.latencies else 0.0,
            "recall_by_rule": {
                rule: round(hit / total, 4) for rule, (hit, total) in sorted(self.per_rule.items()) if total
            },
            "recall_by_novel": {
                novel: round(hit / total, 4) for novel, (hit, total) in sorted(self.per_novel.items()) if total
            },
        }


def score(items, predict):
    """Run a predictor over items and return Scores.

    A predictor carrying a fit() hook is shown the whole item set first, which is
    what lets the corpus-statistics attacks model an adversary who downloaded the
    published dataset rather than one reading a single passage.
    """
    if hasattr(predict, "fit"):
        predict.fit(items)
    scores = Scores()
    for item in items:
        started = time.perf_counter()
        prediction = predict(item)
        scores.latencies.append(time.perf_counter() - started)
        scores.cost_usd += prediction.cost_usd
        scores.input_tokens += prediction.input_tokens
        scores.output_tokens += prediction.output_tokens
        truth = item["ground_truth"]["has_error"]
        if truth:
            rule = item["ground_truth"]["rule"]
            scores.per_rule[rule][1] += 1
            scores.per_novel[item["novel_id"]][1] += 1
            if prediction.has_error:
                scores.true_positive += 1
                scores.per_rule[rule][0] += 1
                scores.per_novel[item["novel_id"]][0] += 1
                if prediction.paragraph_index == item["injected_location"]["paragraph_index"]:
                    scores.localized += 1
            else:
                scores.false_negative += 1
        else:
            if prediction.has_error:
                scores.false_positive += 1
            else:
                scores.true_negative += 1
    return scores


PREDICTORS = {}


HEURISTIC_KINDS = ("attack", "floor", "diagnostic")


def predictor(name, kind="attack", needs_key=False):
    """Register a predictor, or a predictor instance, under a name.

    kind is declared rather than inferred from the name: the leaderboard colours and
    legends runs by it, and a name-prefix rule would publish a future `jev_naive` or
    `qwen3_local` baseline as a "free heuristic".
    """

    def wrap(fn):
        fn.predictor_name = name
        fn.kind = kind
        fn.needs_key = needs_key
        PREDICTORS[name] = fn
        return fn

    return wrap


@predictor("always_clean", kind="floor")
def always_clean(item):
    """Floor: never flag anything. Shows what recall-free perfect specificity scores."""
    return Prediction(has_error=False)


@predictor("always_error", kind="floor")
def always_error(item):
    """Floor: flag everything. On a balanced paired corpus this gives precision 0.5."""
    return Prediction(has_error=True, paragraph_index=0)


def _paragraph_at(text, offset):
    """0-based index of the paragraph holding offset, or None for a failed find().

    The injected label and a predictor's answer are compared for localization, so
    both sides derive the index the same way rather than each counting separators.
    """
    return text.count("\n\n", 0, offset) if offset >= 0 else None


def _locate(text, target):
    """Prediction paragraph index for the first occurrence of target."""
    return _paragraph_at(text, text.find(target))


@predictor("singleton_name")
def singleton_name(item):
    """Attack: flag a passage holding a name mentioned once beside names mentioned often.

    character_swap replaces one mention of a frequent character with an absent one,
    so the intruder appears exactly once. If this scores well, the corpus is leaking
    the injection rather than testing continuity reasoning.
    """
    text = item["text"]
    counts = bench.name_counts(text)
    if not counts:
        return Prediction(has_error=False)
    singletons = [name for name, count in counts.items() if count == 1]
    frequent = [name for name, count in counts.items() if count >= 3]
    if not singletons or not frequent:
        return Prediction(has_error=False)
    target = singletons[0]
    return Prediction(
        has_error=True,
        paragraph_index=_locate(text, target),
        note=f"singleton {target} beside {len(frequent)} frequent names",
    )


@predictor("rarest_name")
def rarest_name(item):
    """Attack: always flag, and localize to the least-mentioned name in the passage."""
    text = item["text"]
    counts = bench.name_counts(text)
    if not counts:
        return Prediction(has_error=True, paragraph_index=0)
    target = min(counts, key=lambda name: (counts[name], name))
    return Prediction(has_error=True, paragraph_index=_locate(text, target))


@predictor("trait_disagreement")
def trait_disagreement(item):
    """Attack: flag when two different colours describe the same feature in one passage.

    Reuses bench.TRAIT_PATTERNS rather than its own colour regex, so the attack cannot
    drift behind the injector: a hand-copied pattern that matched only "her blue eyes"
    would miss the "her eyes were blue" and "blue-eyed" forms bench also injects, and
    would under-report leakage without failing anything.
    """
    text = item["text"]
    offsets = defaultdict(dict)
    for noun, pattern in bench.TRAIT_PATTERNS:
        for m in pattern.finditer(text):
            offsets[noun][m.group("value").lower()] = m.start()
    for noun, by_value in offsets.items():
        values = list(by_value)
        if any(not bench.same_colour(a, b) for a in values for b in values):
            return Prediction(
                has_error=True, paragraph_index=_paragraph_at(text, max(by_value.values()))
            )
    return Prediction(has_error=False)


@predictor("narration_person")
def narration_person(item):
    """Attack: flag a passage whose narration mixes third-person and first-person.

    pov_slip works by putting "I <verb>" into third-person narration, and this is the
    one-line test for exactly that, so it is the honest measure of whether that error
    type is trivial. Run it before believing a pov_slip result.
    """
    text = item["text"]
    spans = bench.quoted_spans(text)
    third = [m for m in bench.THIRD_PERSON_NARRATION.finditer(text) if bench.outside_quotes(spans, m.start())]
    first = [m for m in bench.FIRST_PERSON_NARRATION.finditer(text) if bench.outside_quotes(spans, m.start())]
    if not third or not first:
        return Prediction(has_error=False)
    return Prediction(
        has_error=True,
        paragraph_index=_paragraph_at(text, first[0].start()),
        note=f"{len(first)} first-person against {len(third)} third-person narration verbs",
    )


class CorpusPriorAttack:
    """Attack: pool every passage of a novel, then flag names that are locally rare but globally common.

    This models the adversary that matters for a published dataset: someone who
    downloads the whole corpus rather than reading one passage. character_swap
    inserts a character with at least 20 novel-wide mentions into a scene they are
    absent from, so the intruder is exactly a name that is frequent across the
    novel and appears once here. If this discriminates, the dataset leaks.
    """

    def __init__(self):
        self.novel_counts = defaultdict(Counter)

    def fit(self, items):
        self.novel_counts = defaultdict(Counter)
        for item in items:
            self.novel_counts[item["novel_id"]].update(bench.name_counts(item["text"]))

    def __call__(self, item):
        text = item["text"]
        local = bench.name_counts(text)
        globally = self.novel_counts[item["novel_id"]]
        suspects = [
            name
            for name, count in local.items()
            if count == 1 and globally[name] >= 20 and any(c >= 3 for c in local.values())
        ]
        if not suspects:
            return Prediction(has_error=False)
        target = min(sorted(suspects), key=lambda name: globally[name])
        offset = text.find(target)
        return Prediction(
            has_error=True,
            paragraph_index=text.count("\n\n", 0, offset) if offset >= 0 else None,
            note=f"locally rare, globally frequent: {target}",
        )


predictor("corpus_prior")(CorpusPriorAttack())


class PairLeakAttack:
    """Diagnostic: compare the two texts of a matched pair and localize the edit.

    Not a deployment attack, but it must stay measured: a matched pair is two
    near-identical texts, so anyone holding both halves can find the edit exactly.
    It scores localization 1.000 and guesses which half carries the error by length,
    which is a coin flip - see PairDiffTargetedAttack for the version that does not
    have to guess. bench.one_half_per_pair is the mitigation.
    """

    def __init__(self):
        self.by_pair = defaultdict(list)

    def fit(self, items):
        self.by_pair = defaultdict(list)
        for item in items:
            self.by_pair[item["pair_id"]].append(item["text"])

    def other_half(self, item):
        """The pair's other text, or None when only one half is present."""
        texts = self.by_pair[item["pair_id"]]
        if len(texts) != 2:
            return None
        other = texts[0] if texts[1] == item["text"] else texts[1]
        return None if other == item["text"] else other

    def __call__(self, item):
        other = self.other_half(item)
        if other is None:
            return Prediction(has_error=False)
        return Prediction(
            has_error=len(item["text"]) >= len(other),
            paragraph_index=first_differing_paragraph(item["text"], other),
            note="diffed against the other half of the pair",
        )


def first_differing_paragraph(text, other):
    """Index of the first paragraph that differs between two near-identical texts.

    Compares paragraphs rather than characters: twelve C-speed string comparisons
    instead of a few thousand interpreted ones, and it yields the index directly.
    """
    mine, theirs = text.split("\n\n"), other.split("\n\n")
    return next(
        (i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b), min(len(mine), len(theirs))
    )


def differing_tokens(text, other):
    """The differing run of text between two near-identical strings, both ways round."""
    head = 0
    limit = min(len(text), len(other))
    while head < limit and text[head] == other[head]:
        head += 1
    tail = 0
    while tail < limit - head and text[len(text) - 1 - tail] == other[len(other) - 1 - tail]:
        tail += 1
    return text[head : len(text) - tail].strip(), other[head : len(other) - tail].strip()


class PairDiffTargetedAttack:
    """Diagnostic: the attack that forced one-half-per-pair publishing.

    Diffing a pair hands over the exact token that was changed, which turns the blind
    guess corpus_prior has to make across a whole passage into a single targeted test:
    the injected half is the one whose differing token appears once while its
    counterpart appears often. Measured at J +0.699 with FPR 0.000 on the full paired
    corpus - an order of magnitude past every blind attack - which is why
    bench.one_half_per_pair ships one half of each test pair and no more.

    It stays registered so the mitigation cannot quietly regress: run it against a
    release export and it should collapse to J 0.000 for want of any pair to diff.
    """

    def __init__(self):
        self.leak = PairLeakAttack()

    def fit(self, items):
        self.leak.fit(items)

    def __call__(self, item):
        other = self.leak.other_half(item)
        if other is None:
            return Prediction(has_error=False, note="no pair available - mitigation holding")
        mine, theirs = differing_tokens(item["text"], other)
        if not mine or not theirs:
            return Prediction(has_error=False)
        flagged = (
            bench.name_counts(item["text"]).get(mine, 0) == 1
            and bench.name_counts(other).get(theirs, 0) >= 3
        )
        return Prediction(
            has_error=flagged,
            paragraph_index=first_differing_paragraph(item["text"], other) if flagged else None,
            note=f"pair diff isolated {mine!r} against {theirs!r}",
        )


predictor("pair_diff_targeted", kind="diagnostic")(PairDiffTargetedAttack())


predictor("pair_leak", kind="diagnostic")(PairLeakAttack())


# USD per million tokens, input then output. A wrong rate silently corrupts the cost
# axis, which is this benchmark's headline claim, so an unlisted model reports 0.0 and
# warns rather than guessing - pass --input-rate/--output-rate to supply the real ones.
# Verify these against current pricing pages before publishing any cost number.
MODEL_PRICING_USD_PER_MTOK = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # Jev bills input only; output is free because it generates none.
    "typesafe:jev-latest": (0.042, 0.0),
}

VERDICT_SCHEMA_NAME = "continuity_verdict"
VERDICT_PROPERTIES = {
    "has_error": {"type": "boolean"},
    "paragraph_index": {"type": "integer"},
    "reason": {"type": "string"},
}
VERDICT_JSON_SCHEMA = {
    "type": "object",
    "properties": VERDICT_PROPERTIES,
    "required": list(VERDICT_PROPERTIES),
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You check passages of fiction for internal continuity errors.

A continuity error is a statement that contradicts something established earlier in
the same passage: a character's stated trait changing with no narrative reason, a
character referred to as present who is not in the scene, an impossible sequence of
days, or a fact that reverses without explanation.

These are NOT continuity errors: deliberate ambiguity, a character lying or being
mistaken, figurative language, archaic spelling or punctuation, a narrator withholding
information, or a scene change.

About half the passages you see contain no error at all. Say so when that is the case;
do not invent an error to have something to report.

Answer with has_error, paragraph_index (the 0-based index of the paragraph holding the
contradiction, or -1 when has_error is false) and a one-sentence reason."""


SECONDS_PER_LOCAL_ITEM = 6.4

FATAL_API_CODES = {
    "insufficient_quota",
    "credit_balance_exhausted",
    "invalid_api_key",
    "account_deactivated",
    "model_not_found",
    "permission_denied",
}


class RunAborted(Exception):
    """A failure that will hit every remaining item, so the run should stop now."""


def classify_api_error(exc):
    """Return a fatal message for an error that will not fix itself, else None.

    A quota or auth failure affects every remaining item, so retrying 400 more times
    wastes wall clock and prints 400 tracebacks. Anything else is treated as transient
    and skipped, so one flaky response cannot throw away a long run.
    """
    text = str(exc)
    body = getattr(exc, "body", None)
    code = ""
    if isinstance(body, dict):
        error = body.get("error") or {}
        code = str(error.get("code") or error.get("type") or "")
    for candidate in FATAL_API_CODES:
        if candidate in code or candidate in text:
            return f"{type(exc).__name__}: {candidate}"
    status = getattr(exc, "status_code", None)
    if status in (401, 403, 404):
        return f"{type(exc).__name__}: HTTP {status}"
    return None


def read_key_file(path):
    """Read an API key from a file, so a key never has to be pasted into a session.

    Returns None when the file is absent, which leaves the SDK to resolve credentials
    from the environment as usual.
    """
    path = Path(path).expanduser()
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip() or None


class VerdictPredictor:
    """Shared prompt, pricing and verdict parsing for any chat model.

    Subclasses implement complete() and nothing else. Keeping one prompt and one
    schema across providers is what makes a cross-provider comparison mean anything -
    two providers judged on two different prompts is not a benchmark.

    ponytail: one call per passage, no prompt caching. The passage is most of the
    prompt and changes every call, so a cached prefix would buy almost nothing here;
    revisit if a per-novel story state gets prepended.
    """

    kind = "model"
    needs_key = True
    provider = "unset"

    sampling = {}

    def __init__(self, model, effort="high", max_tokens=2000, rates=None):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.rates = rates
        self.predictor_name = f"{self.provider}_{model}_{effort}"
        self._client = None
        self.parse_failures = 0
        self.parse_failure_notes = []
        self.api_failures = 0
        self.api_failure_notes = []

    def client(self):
        """The provider SDK client, constructed on first use."""
        if self._client is None:
            self._client = self.build_client()
        return self._client

    def build_client(self):
        raise NotImplementedError

    def complete(self, prompt):
        """Return (text, input_tokens, output_tokens) for one passage."""
        raise NotImplementedError

    def price(self, input_tokens, output_tokens):
        """Cost in USD for one call, or 0.0 when no rate is known for the model."""
        rates = self.rates or MODEL_PRICING_USD_PER_MTOK.get(self.model)
        if rates is None:
            return 0.0
        return input_tokens / 1e6 * rates[0] + output_tokens / 1e6 * rates[1]

    def has_rates(self):
        return bool(self.rates or MODEL_PRICING_USD_PER_MTOK.get(self.model))

    def build_prompt(self, item):
        """Number the paragraphs so the model can point at one."""
        numbered = "\n\n".join(
            f"[{index}] {para}" for index, para in enumerate(item["text"].split("\n\n"))
        )
        return f"Passage:\n\n{numbered}\n\nDoes this passage contradict itself?"

    def __call__(self, item):
        try:
            text, input_tokens, output_tokens = self.complete(self.build_prompt(item))
        except Exception as exc:
            fatal = classify_api_error(exc)
            if fatal:
                raise RunAborted(fatal) from exc
            self.api_failures += 1
            self.api_failure_notes.append(f"{type(exc).__name__}: {str(exc)[:160]}")
            return Prediction(has_error=False, note=f"API FAILURE: {type(exc).__name__}")
        cost = self.price(input_tokens, output_tokens)
        try:
            verdict = json.loads(text)
        except json.JSONDecodeError:
            self.parse_failures += 1
            self.parse_failure_notes.append(text[:200])
            return Prediction(
                has_error=False,
                cost_usd=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                note="PARSE FAILURE: " + text[:200],
            )
        paragraph = verdict.get("paragraph_index")
        return Prediction(
            has_error=bool(verdict.get("has_error")),
            paragraph_index=paragraph if isinstance(paragraph, int) and paragraph >= 0 else None,
            cost_usd=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            note=str(verdict.get("reason", ""))[:300],
        )


class AnthropicPredictor(VerdictPredictor):
    """Ask a Claude model whether a passage contradicts itself.

    Credentials: ANTHROPIC_API_KEY, an `ant auth login` profile the SDK finds on its
    own, or ~/.anthropic-key.
    """

    provider = "claude"

    def build_client(self):
        import anthropic

        key = read_key_file("~/.anthropic-key")
        return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()

    def complete(self, prompt):
        response = self.client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": VERDICT_JSON_SCHEMA},
            },
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((block.text for block in response.content if block.type == "text"), "")
        return text, response.usage.input_tokens, response.usage.output_tokens


class OpenAIPredictor(VerdictPredictor):
    """Ask an OpenAI model the same question, with the same prompt and schema.

    Credentials: OPENAI_API_KEY or ~/.openai-key. Uses Chat Completions with a strict
    json_schema response format so the verdict parses without prompt-wrangling.
    """

    provider = "openai"

    def build_client(self):
        from openai import OpenAI

        key = read_key_file("~/.openai-key")
        return OpenAI(api_key=key) if key else OpenAI()

    def complete(self, prompt):
        response = self.client().chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            **self.sampling,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": VERDICT_SCHEMA_NAME,
                    "strict": True,
                    "schema": VERDICT_JSON_SCHEMA,
                },
            },
        )
        usage = response.usage
        return (
            response.choices[0].message.content or "",
            usage.prompt_tokens,
            usage.completion_tokens,
        )


class LocalPredictor(OpenAIPredictor):
    """Ask a model served locally by Ollama the same question, over its OpenAI-compatible API.

    Ollama's /v1 endpoint accepts and honours a strict json_schema response_format, so
    this needs nothing but a base URL: the prompt, schema and verdict parsing are the
    same ones the hosted providers get, which is the only way the comparison means
    anything. Satisfies the brief's open-LLM baseline.

    API cost is genuinely zero, not merely unknown, so rates are pinned to (0, 0)
    rather than left to the unknown-model warning. That is marginal API cost only -
    local inference costs wall clock and hardware instead, which is why the results
    record median latency alongside it.
    """

    provider = "local"
    needs_key = False
    # Greedy decoding with a fixed seed. Without this the same 40 items scored J -0.100
    # and then J +0.250 on consecutive runs of the same model - a 0.35 swing that would
    # make any leaderboard position meaningless. Hosted frontier models do not accept
    # sampling parameters at all, which is part of why every run reports a confidence
    # interval rather than a bare point estimate.
    sampling = {"temperature": 0.0, "seed": 20260922}

    def __init__(self, model, base_url="http://localhost:11434/v1", **kw):
        kw.setdefault("rates", (0.0, 0.0))
        super().__init__(model=model, **kw)
        self.base_url = base_url

    def build_client(self):
        from openai import OpenAI

        return OpenAI(base_url=self.base_url, api_key="ollama")



predictor("llm", kind="model", needs_key=True)(AnthropicPredictor("claude-opus-5"))
predictor("openai", kind="model", needs_key=True)(OpenAIPredictor("gpt-4.1"))
predictor("local", kind="model", needs_key=False)(LocalPredictor("llama3.1:8b"))

class JevPredictor(VerdictPredictor):
    """The naive Jev-only pass: one typed question per passage, no LLM anywhere.

    The brief asks for this baseline explicitly and asks for it reported honestly -
    it is where Jev alone is expected to fall short, because a bare passage is not the
    story state that a System One model is meant to be checking against. JevHybrid is
    the configuration the project is actually proposing.

    Jev returns a calibrated probability rather than sampled text, so the confidence
    is recorded on every prediction and is what the hybrid thresholds on.
    """

    provider = "jev"
    needs_key = True

    def __init__(self, model="typesafe:jev-latest", **kw):
        kw.setdefault("rates", MODEL_PRICING_USD_PER_MTOK.get(model))
        super().__init__(model=model, **kw)
        self._verifier = None

    def verifier(self):
        if self._verifier is None:
            import pipeline

            self._verifier = pipeline.JevVerifier(self.model)
        return self._verifier

    def __call__(self, item):
        import pipeline

        verifier = self.verifier()
        before = verifier.input_tokens
        state = pipeline.StoryState(
            established_facts=["(no separate state: the passage is checked against itself)"]
        )
        try:
            contradicts, confidence = verifier.check(state, item["text"])
        except Exception as exc:
            fatal = classify_api_error(exc)
            if fatal:
                raise RunAborted(fatal) from exc
            self.api_failures += 1
            self.api_failure_notes.append(f"{type(exc).__name__}: {str(exc)[:160]}")
            return Prediction(has_error=False, note=f"API FAILURE: {type(exc).__name__}")
        used = verifier.input_tokens - before
        return Prediction(
            has_error=contradicts,
            cost_usd=self.price(used, 0),
            confidence=confidence,
            input_tokens=used,
            note=f"jev confidence {confidence:.3f}",
        )


class JevHybridPredictor(VerdictPredictor):
    """The proposal itself: LLM extracts state, Jev checks every paragraph, LLM confirms.

    This is the run the leaderboard leads with, because it is the claim - match
    full-context LLM accuracy at a fraction of the cost. Cost here is honest about the
    extraction call, which on a real manuscript amortises across a whole chapter but on
    a single benchmark passage does not, so this number is an upper bound on what the
    pipeline costs in production.
    """

    provider = "jev_hybrid"
    needs_key = True

    def __init__(self, model="typesafe:jev-latest", helper_model="claude-opus-5", **kw):
        kw.setdefault("rates", MODEL_PRICING_USD_PER_MTOK.get(model))
        super().__init__(model=model, **kw)
        self.helper_model = helper_model
        self.predictor_name = f"jev_hybrid_{helper_model}"
        self._verifier = None
        self._helper = None
        self.escalations = 0
        self.paragraphs_checked = 0

    def verifier(self):
        if self._verifier is None:
            import pipeline

            self._verifier = pipeline.JevVerifier(self.model)
        return self._verifier

    def helper(self):
        if self._helper is None:
            self._helper = AnthropicPredictor(self.helper_model)
        return self._helper

    def extract(self, text):
        import pipeline

        helper = self.helper()
        raw, tokens_in, tokens_out = helper.complete(
            pipeline.EXTRACTION_PROMPT.format(text=text)
        )
        self.helper_cost = getattr(self, "helper_cost", 0.0) + helper.price(tokens_in, tokens_out)
        return pipeline.parse_state(raw)

    def escalate(self, state, index, paragraph, text):
        import pipeline

        helper = self.helper()
        raw, tokens_in, tokens_out = helper.complete(
            pipeline.ESCALATION_PROMPT.format(
                state=state.as_prompt(), index=index, paragraph=paragraph, text=text
            )
        )
        self.helper_cost = getattr(self, "helper_cost", 0.0) + helper.price(tokens_in, tokens_out)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self.parse_failures += 1
            return None
        return pipeline.EscalationVerdict(
            is_real_contradiction=bool(payload.get("is_real_contradiction")),
            paragraph_index=int(payload.get("paragraph_index", index)),
            explanation=str(payload.get("explanation", "")),
        )

    def __call__(self, item):
        import pipeline

        self.helper_cost = getattr(self, "helper_cost", 0.0)
        before_cost = self.helper_cost
        verifier = self.verifier()
        before_tokens = verifier.input_tokens
        try:
            result = pipeline.run_pipeline(
                item["text"], verifier, self.extract, self.escalate
            )
        except Exception as exc:
            fatal = classify_api_error(exc)
            if fatal:
                raise RunAborted(fatal) from exc
            self.api_failures += 1
            self.api_failure_notes.append(f"{type(exc).__name__}: {str(exc)[:160]}")
            return Prediction(has_error=False, note=f"API FAILURE: {type(exc).__name__}")
        self.escalations += result.escalations
        self.paragraphs_checked += result.paragraphs_checked
        confirmed = result.confirmed
        jev_tokens = verifier.input_tokens - before_tokens
        return Prediction(
            has_error=bool(confirmed),
            paragraph_index=confirmed[0].paragraph_index if confirmed else None,
            cost_usd=self.price(jev_tokens, 0) + (self.helper_cost - before_cost),
            confidence=confirmed[0].jev_confidence if confirmed else None,
            input_tokens=jev_tokens,
            note=confirmed[0].explanation[:300] if confirmed else "no confirmed contradiction",
        )


predictor("jev", kind="model", needs_key=True)(JevPredictor())
predictor("jev_hybrid", kind="model", needs_key=True)(JevHybridPredictor())

class HybridLocalPredictor(VerdictPredictor):
    """The Jev architecture with a local stand-in in Jev's slot, runnable today.

    Same three stages as JevHybridPredictor - extract state, gate every paragraph,
    escalate only what the gate flags - but the gate and the escalator are both local
    models, so it needs no keys and no credits. Swapping the gate back to JevVerifier
    is one line.

    What this run can establish: that the harness works on real corpus items, what the
    escalation rate is, and whether gate-plus-escalation beats a single whole-passage
    pass by the same model. What it cannot establish: anything about Jev's accuracy,
    its calibration, or the cost claim - a local gate costs a model call per paragraph,
    which is the cost Jev exists to remove. `projected_jev_gate_usd` in the results
    applies TypeSafe's published rate to the measured gate tokens and is labelled a
    projection for that reason.
    """

    provider = "hybrid_local"
    needs_key = False

    def __init__(self, model="llama3.1:8b", gate_model=None, **kw):
        kw.setdefault("rates", (0.0, 0.0))
        super().__init__(model=model, **kw)
        self.gate_model = gate_model or model
        self.predictor_name = f"hybrid_local_{self.gate_model}"
        self._verifier = None
        self._helper = None
        self.escalations = 0
        self.paragraphs_checked = 0
        self.gate_input_tokens = 0

    def verifier(self):
        if self._verifier is None:
            import pipeline

            self._verifier = pipeline.LocalVerifier(self.gate_model)
        return self._verifier

    def helper(self):
        if self._helper is None:
            self._helper = LocalPredictor(self.model)
        return self._helper

    def extract(self, text):
        import pipeline

        raw, _, _ = self.helper().complete(pipeline.EXTRACTION_PROMPT.format(text=text))
        return pipeline.parse_state(raw)

    def escalate(self, state, index, paragraph, text):
        import pipeline

        raw, _, _ = self.helper().complete(
            pipeline.ESCALATION_PROMPT.format(
                state=state.as_prompt(), index=index, paragraph=paragraph, text=text
            )
        )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self.parse_failures += 1
            return None
        return pipeline.EscalationVerdict(
            is_real_contradiction=bool(payload.get("is_real_contradiction")),
            paragraph_index=int(payload.get("paragraph_index", index)),
            explanation=str(payload.get("explanation", "")),
        )

    def __call__(self, item):
        import pipeline

        verifier = self.verifier()
        before = verifier.input_tokens
        try:
            result = pipeline.run_pipeline(item["text"], verifier, self.extract, self.escalate)
        except Exception as exc:
            if classify_api_error(exc):
                raise RunAborted(classify_api_error(exc)) from exc
            self.api_failures += 1
            self.api_failure_notes.append(f"{type(exc).__name__}: {str(exc)[:160]}")
            return Prediction(has_error=False, note=f"API FAILURE: {type(exc).__name__}")
        self.escalations += result.escalations
        self.paragraphs_checked += result.paragraphs_checked
        self.gate_input_tokens += verifier.input_tokens - before
        confirmed = result.confirmed
        return Prediction(
            has_error=bool(confirmed),
            paragraph_index=confirmed[0].paragraph_index if confirmed else None,
            cost_usd=0.0,
            confidence=confirmed[0].jev_confidence if confirmed else None,
            input_tokens=verifier.input_tokens - before,
            note=confirmed[0].explanation[:300] if confirmed else "no confirmed contradiction",
        )


predictor("hybrid_local", kind="model", needs_key=False)(HybridLocalPredictor())



PROVIDERS = {
    "llm": AnthropicPredictor,
    "openai": OpenAIPredictor,
    "local": LocalPredictor,
    "jev": JevPredictor,
    "jev_hybrid": JevHybridPredictor,
    "hybrid_local": HybridLocalPredictor,
}
DEFAULT_MODELS = {
    "llm": "claude-opus-5",
    "openai": "gpt-4.1",
    "local": "llama3.1:8b",
    "jev": "typesafe:jev-latest",
    "jev_hybrid": "typesafe:jev-latest",
    "hybrid_local": "llama3.1:8b",
}



def load_items(corpus, split=None, rule=None, limit=None):
    """Read corpus items, optionally filtered, keeping injected/clean pairs together."""
    items = []
    for line in Path(corpus).open(encoding="utf-8"):
        item = json.loads(line)
        if split and item["split"] != split:
            continue
        items.append(item)
    if rule:
        keep = {
            item["pair_id"] for item in items if item["ground_truth"]["rule"] == rule
        }
        items = [item for item in items if item["pair_id"] in keep]
    if limit:
        keep = set(list(dict.fromkeys(item["pair_id"] for item in items))[:limit])
        items = [item for item in items if item["pair_id"] in keep]
    return items


def write_results(name, corpus, items, scores, extra=None, kind="attack"):
    """Write one predictor's results to results/<name>.json and return the path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "predictor": name,
        "kind": kind,
        "corpus": str(Path(corpus).name),
        "items": len(items),
        "pairs": len({item["pair_id"] for item in items}),
        "novels": len({item["novel_id"] for item in items}),
        **(extra or {}),
        **scores.summary(),
    }
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path, payload


def print_row(name, summary):
    lo, hi = summary["youden_j_ci95"]
    mark = {
        "effect": "",
        "no_effect": "  (no discrimination, tight CI)",
        "inconclusive": "  (inconclusive, CI too wide)",
    }[summary["youden_j_verdict"]]
    print(
        f"{name:22} J {summary['youden_j']:+.3f} [{lo:+.3f},{hi:+.3f}]  "
        f"F1 {summary['f1']:.3f}  R {summary['recall']:.3f}  "
        f"FPR {summary['false_positive_rate']:.3f}  loc {summary['localization']:.3f}  "
        f"${summary['cost_usd']:.4f}{mark}"
    )


def cmd_run(args):
    if args.predictor not in PREDICTORS:
        print(f"unknown predictor {args.predictor!r}; try `evaluate.py list`")
        return 2
    items = load_items(args.corpus, args.split, args.rule, args.limit)
    if not items:
        print("no items matched the filters")
        return 2
    predict = PREDICTORS[args.predictor]
    if args.predictor in PROVIDERS:
        rates = None
        if args.input_rate is not None and args.output_rate is not None:
            rates = (args.input_rate, args.output_rate)
        kwargs = {"model": args.model or DEFAULT_MODELS[args.predictor], "effort": args.effort}
        if rates is not None:
            kwargs["rates"] = rates
        if args.predictor == "local" and args.base_url:
            kwargs["base_url"] = args.base_url
        predict = PROVIDERS[args.predictor](**kwargs)
    name = getattr(predict, "predictor_name", args.predictor)
    if predict.kind == "model" and not predict.needs_key:
        minutes = len(items) * SECONDS_PER_LOCAL_ITEM / 60
        print(
            f"{name}: {len(items)} items against a local model, roughly {minutes:.0f} min. "
            "No API cost, but it is not instant."
        )
    if predict.needs_key:
        pairs = len({item["pair_id"] for item in items})
        print(f"{name}: {len(items)} items ({pairs} pairs) against a paid API.")
        if not predict.has_rates():
            print(
                f"WARNING no published rate known for {predict.model!r}, so cost will be "
                "reported as $0.00. Pass --input-rate and --output-rate (USD per million "
                "tokens) to record the real cost."
            )
        if not args.yes:
            print("This spends real money. Re-run with --yes to confirm.")
            return 2
    try:
        scores = score(items, predict)
    except RunAborted as exc:
        print(f"\nABORTED after the API returned a failure that affects every item: {exc}")
        if "quota" in str(exc) or "credit" in str(exc):
            print("Add credits, then re-run. Nothing was scored and no results file was written.")
        return 3
    extra = {
        key: value
        for key in ("model", "effort")
        if (value := getattr(predict, key, None)) is not None
    }
    import pipeline as _pipeline

    if getattr(predict, "paragraphs_checked", 0):
        extra["paragraphs_checked"] = predict.paragraphs_checked
        extra["escalations"] = predict.escalations
        extra["escalation_rate"] = round(
            predict.escalations / predict.paragraphs_checked, 4
        )
    if getattr(predict, "gate_input_tokens", 0):
        extra["gate_input_tokens"] = predict.gate_input_tokens
        extra["projected_jev_gate_usd"] = round(
            _pipeline.project_jev_cost(predict.gate_input_tokens), 6
        )
    failures = getattr(predict, "parse_failures", 0)
    api_failures = getattr(predict, "api_failures", 0)
    if failures:
        extra["parse_failures"] = failures
    if api_failures:
        extra["api_failures"] = api_failures
    path, payload = write_results(name, args.corpus, items, scores, extra, predict.kind)
    print_row(name, payload)
    print(f"by rule: {payload['recall_by_rule']}")
    if failures:
        print(f"WARNING {failures} response(s) failed to parse and were counted as 'no error'")
        for note in getattr(predict, "parse_failure_notes", [])[:3]:
            print(f"  {note}")
    if api_failures:
        print(
            f"WARNING {api_failures} item(s) hit a transient API failure, were skipped, and "
            "count against recall - treat this run as incomplete"
        )
        for note in getattr(predict, "api_failure_notes", [])[:3]:
            print(f"  {note}")
    print(f"wrote {path}")
    return 0


def cmd_attack(args):
    items = load_items(args.corpus, args.split, args.rule, args.limit)
    if not items:
        print("no items matched the filters")
        return 2
    print(f"{len({i['pair_id'] for i in items})} pairs, {len(items)} items\n")
    best, winner = -1.0, None
    for name, predict in PREDICTORS.items():
        if predict.kind not in HEURISTIC_KINDS:
            continue
        scores = score(items, predict)
        _, payload = write_results(name, args.corpus, items, scores, kind=predict.kind)
        print_row(name, payload)
        print(f"{'':22} by rule: {payload['recall_by_rule']}")
        if predict.kind != "diagnostic" and payload["youden_j"] > best:
            best, winner = payload["youden_j"], name
    print(
        "\nNote: flagging every passage scores F1 0.667 on a balanced paired corpus while "
        "discriminating nothing, so leakage is judged on J = recall - FPR, not on F1."
    )
    print(f"best zero-cost J: {best:+.3f} ({winner}), excluding diagnostics")
    if best >= args.threshold:
        print(
            f"VERDICT: a free heuristic reaches J {best:+.3f} >= {args.threshold}. "
            "The corpus leaks its injection method and is not ready for a paid eval."
        )
        return 1
    print(f"VERDICT: no free heuristic reaches J {args.threshold}. Discrimination is not leaked.")
    return 0


def cmd_list(args):
    for name, predict in PREDICTORS.items():
        key = " (needs API key)" if predict.needs_key else ""
        doc = (predict.__doc__ or "").strip().splitlines()[0]
        doc = doc if len(doc) < 100 else doc[:97] + "..."
        print(f"{name:22} [{predict.kind}]{key}  {doc}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
        p.add_argument("--split", choices=("dev", "test"))
        p.add_argument("--rule", help="keep only pairs whose injected item used this rule")
        p.add_argument("--limit", type=int, help="first N pairs only")

    run = sub.add_parser("run", help="score one predictor")
    run.add_argument("predictor")
    common(run)
    run.add_argument("--model", help="model id, for a provider predictor")
    run.add_argument("--effort", default="high", help="reasoning effort, where the provider has one")
    run.add_argument("--base-url", help="OpenAI-compatible endpoint, for the local predictor")
    run.add_argument("--input-rate", type=float, help="USD per million input tokens")
    run.add_argument("--output-rate", type=float, help="USD per million output tokens")
    run.add_argument("--yes", action="store_true", help="confirm spending on a paid predictor")
    run.set_defaults(func=cmd_run)

    attack = sub.add_parser("attack", help="run every zero-cost heuristic against the corpus")
    common(attack)
    attack.add_argument("--threshold", type=float, default=0.3, help="J that counts as leakage")
    attack.set_defaults(func=cmd_attack)

    listing = sub.add_parser("list", help="show registered predictors")
    listing.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
