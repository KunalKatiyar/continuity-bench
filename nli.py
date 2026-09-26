#!/usr/bin/env python3
"""Continuity checking framed as natural language inference.

A contradiction between two paragraphs is an NLI problem: premise = what the story
already established, hypothesis = the new paragraph, label = contradiction. That
reduction matters because it means an encoder trained on MNLI/ANLI can attempt this
task with no prompting, no generation and no API, and because it is the framing a
fine-tuned gate would also use.

Scoring is all-pairs within a passage: for every ordered pair of paragraphs (i, j) with
i < j, ask P(paragraph j contradicts paragraph i). The passage score is the maximum over
pairs, and the paragraph it points at is the localization. That is a direct statement of
the task - does any paragraph contradict any earlier one - rather than an approximation
of it, and it sidesteps the 512-token limit that truncating a growing premise would hit.

ponytail: all-pairs is O(n^2) in paragraphs, which is 66 pairs for a 12-paragraph
passage and entirely affordable. On a whole manuscript it would not be; there the story
state is the premise, which is the point of the pipeline and why this file is a baseline
rather than the product.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
ROOT = Path(__file__).resolve().parent


class NLIScorer:
    """Wraps an MNLI-style cross-encoder and returns P(contradiction) for text pairs."""

    def __init__(self, model_name=DEFAULT_MODEL, batch_size=16, max_length=512, device=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._model = None
        self._tokenizer = None
        self.contradiction_index = None
        self.pairs_scored = 0
        self.forward_passes = 0
        self.hypotheses_clipped = 0
        self._clipped = {}

    def load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self.device).eval()
        self.contradiction_index = self._find_contradiction_label()

    def _find_contradiction_label(self):
        """Locate the contradiction logit by name rather than assuming an index.

        MNLI checkpoints disagree on label order, and guessing it silently inverts the
        whole benchmark, so the label map is read from the model config.
        """
        labels = getattr(self._model.config, "id2label", {}) or {}
        for index, name in labels.items():
            if str(name).lower().startswith("contradict"):
                return int(index)
        raise ValueError(f"no contradiction label in {labels!r} for {self.model_name}")

    def clip_hypothesis(self, text):
        """Cap the judged paragraph at half the budget so the premise always has room.

        truncation="only_first" raises outright when the hypothesis alone exceeds
        max_length, and paragraphs here run to 1,831 tokens against a median of 67.
        Clipping the hypothesis explicitly, rather than letting longest_first decide,
        keeps the cut predictable: the premise is context and can be shortened, the
        paragraph being judged is the evidence the label depends on.
        """
        if text in self._clipped:
            return self._clipped[text]
        budget = self.max_length // 2
        ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= budget:
            self._clipped[text] = text
        else:
            self.hypotheses_clipped += 1
            self._clipped[text] = self._tokenizer.decode(ids[:budget], skip_special_tokens=True)
        return self._clipped[text]

    def score_pairs(self, pairs):
        """Return P(contradiction) for a list of (premise, hypothesis) tuples.

        Identical pairs are scored once. Both halves of a matched pair differ in a
        single paragraph, so most of their paragraph pairs are the same two strings,
        and a run repeats ~42% of its forward passes without it. Scoring is a pure
        function of the pair, so the memo changes no result.

        Pairs are also sorted by length before batching, which keeps padding close to
        the real token count instead of padding every batch to its own longest row.
        """
        import torch

        self.load()
        pairs = [(premise, self.clip_hypothesis(hypothesis)) for premise, hypothesis in pairs]
        unique = list(dict.fromkeys(pairs))
        order = sorted(range(len(unique)), key=lambda i: len(unique[i][0]) + len(unique[i][1]))
        scored = {}
        for start in range(0, len(order), self.batch_size):
            batch = [unique[i] for i in order[start : start + self.batch_size]]
            encoded = self._tokenizer(
                [p for p, _ in batch],
                [h for _, h in batch],
                return_tensors="pt",
                # only_first cuts the premise, never the hypothesis. With longest_first
                # the paragraph being judged is the one that gets clipped on 22% of
                # pairs, which silently costs exactly the evidence the label depends on.
                truncation="only_first",
                max_length=self.max_length,
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[:, self.contradiction_index]
            scored.update(zip(batch, probabilities.tolist()))
        self.pairs_scored += len(pairs)
        self.forward_passes += -(-len(unique) // self.batch_size)
        return [scored[pair] for pair in pairs]


def paragraph_pairs(text, max_paragraphs=16):
    """Every ordered pair (earlier, later) of paragraphs in a passage."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()][:max_paragraphs]
    return [
        (paragraphs[i], paragraphs[j], j)
        for j in range(1, len(paragraphs))
        for i in range(j)
    ]


def score_passage(scorer, text):
    """Return (max contradiction probability, paragraph index it points at)."""
    triples = paragraph_pairs(text)
    if not triples:
        return 0.0, None
    scores = scorer.score_pairs([(premise, hypothesis) for premise, hypothesis, _ in triples])
    best = max(range(len(scores)), key=scores.__getitem__)
    return scores[best], triples[best][2]


def sweep_thresholds(scored):
    """Best J over every threshold the data can actually distinguish, plus the curve.

    Candidate cuts are the observed scores, not a fixed grid. A grid is both wasteful
    and wrong here: an MNLI contradiction softmax saturates, and on this corpus the
    median score is 0.986 with 45% of items above 0.99 - so a grid topping out at 0.99
    cannot separate nearly half the data and reports a cut that is an artefact of where
    the grid stopped.

    Metrics come from evaluate.Scores so there is one definition of recall, FPR and J
    in the project rather than a second copy that can drift.
    """
    curve = []
    for threshold in sorted({probability for probability, _ in scored}):
        counts = evaluate.Scores()
        for probability, truth in scored:
            flagged = probability >= threshold
            if truth and flagged:
                counts.true_positive += 1
            elif truth:
                counts.false_negative += 1
            elif flagged:
                counts.false_positive += 1
            else:
                counts.true_negative += 1
        low, high = counts.youden_j_interval
        curve.append({
            "threshold": round(threshold, 6),
            "recall": round(counts.recall, 4),
            "fpr": round(counts.false_positive_rate, 4),
            "youden_j": round(counts.youden_j, 4),
            "youden_j_ci95": [round(low, 4), round(high, 4)],
            "verdict": evaluate.j_verdict(low, high),
        })
    best = max(curve, key=lambda row: row["youden_j"])
    return best, curve


def roc_auc(scored):
    """Rank-based AUC: the probability a random injected item outranks a random clean one.

    Threshold-free, so it says whether the model separates the two classes at all,
    independently of where a decision boundary is drawn.
    """
    positives = [p for p, truth in scored if truth]
    negatives = [p for p, truth in scored if not truth]
    if not positives or not negatives:
        return 0.0
    ranked = sorted(scored, key=lambda row: row[0])
    rank_sum, index = 0.0, 0
    while index < len(ranked):
        end = index
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (index + end + 1) / 2
        rank_sum += sum(average_rank for _, truth in ranked[index:end] if truth)
        index = end
    return (rank_sum - len(positives) * (len(positives) + 1) / 2) / (
        len(positives) * len(negatives)
    )


def per_rule_auc(records):
    """AUC for each rule, scoring its injected items against their own matched controls.

    The headline AUC mixes error types, which hides the question that matters: whether
    the NLI framing fits some kinds of continuity error and not others. A rule whose
    "error" is not a logical contradiction - a character appearing in a scene they are
    absent from is a presupposition failure, not a contradiction - should sit at 0.5
    here even if the framing works elsewhere.
    """
    by_pair = {}
    for record in records:
        by_pair.setdefault(record["pair_id"], []).append(record)
    out = {}
    for rule in sorted({r["rule"] for r in records if r["rule"]}):
        scored = []
        for group in by_pair.values():
            if not any(r["rule"] == rule for r in group):
                continue
            for record in group:
                scored.append((record["probability"], record["has_error"]))
        if scored:
            out[rule] = {
                "auc": round(roc_auc(scored), 4),
                "items": len(scored),
                "best_j": sweep_thresholds(scored)[0]["youden_j"] if len(scored) > 3 else None,
            }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--corpus", default=str(ROOT / "corpus" / "continuity_v0.jsonl"))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--rule", help="keep only pairs whose injected item used this rule")
    parser.add_argument("--limit", type=int, help="first N pairs only")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--scores", default=str(ROOT / "nli_scores.json"))
    args = parser.parse_args(argv)

    items = evaluate.load_items(args.corpus, args.split, args.rule, args.limit)
    scorer = NLIScorer(args.model, batch_size=args.batch_size, max_length=args.max_length)
    scorer.load()
    print(f"{args.model} on {scorer.device}, {len(items)} items")

    records, probabilities = [], {}
    for n, item in enumerate(items, 1):
        probability, paragraph = score_passage(scorer, item["text"])
        probabilities[item["passage_id"]] = (probability, paragraph)
        records.append({
            "passage_id": item["passage_id"],
            "pair_id": item["pair_id"],
            "probability": probability,
            "paragraph_index": paragraph,
            "has_error": item["ground_truth"]["has_error"],
            "truth_paragraph": (item["injected_location"] or {}).get("paragraph_index"),
            "rule": item["ground_truth"]["rule"],
        })
        if n % 50 == 0:
            print(f"  {n}/{len(items)} items, {scorer.pairs_scored} pairs")

    scored = [(r["probability"], r["has_error"]) for r in records]
    best, curve = sweep_thresholds(scored)
    auc = roc_auc(scored)
    by_rule = per_rule_auc(records)
    # The leaderboard row uses the neutral 0.5 cut, not the swept best. A threshold
    # fitted on the split it is scored on is an upper bound, and publishing it in the
    # same J column as unfitted runs is the mislabelling `kind` exists to prevent.
    threshold = 0.5

    # the run goes through the shared scorer so it lands on the leaderboard like any
    # other approach, rather than being a number that only exists in a log
    def predict(item):
        probability, paragraph = probabilities[item["passage_id"]]
        return evaluate.Prediction(
            has_error=probability >= threshold,
            paragraph_index=paragraph,
            confidence=probability,
            note=f"P(contradiction)={probability:.4f}",
        )

    name = f"nli_{args.model.split('/')[-1]}"
    scores = evaluate.score(items, predict)
    path, payload = evaluate.write_results(
        name, args.corpus, items, scores,
        extra={
            "model": args.model,
            "threshold": threshold,
            "best_threshold_fitted": best["threshold"],
            "best_threshold_j_upper_bound": best["youden_j"],
            "per_rule_auc": by_rule,
        },
        kind="model",
    )
    Path(args.scores).write_text(
        json.dumps({"model": args.model, "split": args.split, "threshold": threshold,
                    "curve": curve, "records": records}, indent=2), encoding="utf-8")

    evaluate.print_row(name, payload)
    print(f"\nAUC {auc:.4f}   (0.5 = no separation at any threshold, and unlike J it "
          f"needs no threshold)")
    print(f"swept best J {best['youden_j']:+.4f} at p>={best['threshold']:.4f}, but that "
          f"threshold is fitted on {args.split} so it is an upper bound, not the row above")
    print("by rule (each scored against its own matched controls):")
    for rule, stats in by_rule.items():
        print(f"  {rule:20} AUC {stats['auc']:.4f}  best-J {stats['best_j']}  n={stats['items']}")
    print(f"\nwrote {path} and {args.scores} "
          f"({scorer.pairs_scored} pairs, {scorer.forward_passes} forward passes, "
          f"{scorer.hypotheses_clipped} hypotheses clipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
