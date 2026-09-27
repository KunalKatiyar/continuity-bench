"""Regression guards for the two things that quietly ruin this benchmark.

The corpus: a rule that grows to dominate makes every other rule statistically
powerless, and the headline number becomes one error type's number wearing the
corpus's name. That already happened once, at 98% character_swap.

The harness: a scoring change that silently stops detecting an obvious contradiction,
or that lets a constant strategy look competent, invalidates every row at once. These
pin the floors that must never move.

These read the built corpus, so they are skipped rather than failed when it is absent -
a fresh clone has no corpus until `bench.py build` runs.
"""

import json
from collections import Counter
from functools import cache
from pathlib import Path

import _selftest
import bench

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus" / "continuity_v0.jsonl"
# the build default plus slack, taken from bench so raising the cap there cannot
# silently disarm this guard
MAX_RULE_SHARE = bench.DEFAULT_MAX_RULE_SHARE + 0.05
MIN_RULES_WITH_POWER = 2
MIN_ITEMS_FOR_POWER = 30


def _evaluate():
    """The harness module, or a skip - it needs the venv, the corpus guards do not."""
    try:
        import evaluate
    except ImportError as exc:
        raise _selftest.Skipped(f"needs the venv: {exc}") from exc
    return evaluate


@cache
def _corpus():
    """The built corpus, or a skip - never a silent pass.

    run_all_checks.sh runs the suites before bench.py build, so on a fresh clone these
    guards have nothing to read. Returning quietly would report ok having checked
    nothing, which is the failure this file exists to prevent.
    """
    if not CORPUS.exists():
        raise _selftest.Skipped("no corpus yet; run bench.py build")
    return tuple(json.loads(line) for line in CORPUS.open(encoding="utf-8"))


def _injected(rows):
    return [r for r in rows if r["ground_truth"]["has_error"]]


def test_no_error_type_dominates_the_corpus():
    """One rule at 98% is how every other rule ends up with n=3 and no power."""
    rows = _corpus()
    counts = Counter(r["ground_truth"]["rule"] for r in _injected(rows))
    total = sum(counts.values())
    worst, n = counts.most_common(1)[0]
    assert n / total <= MAX_RULE_SHARE, (
        f"{worst} is {n / total:.0%} of injected items (max {MAX_RULE_SHARE:.0%}); "
        "rebuild with --max-rule-share"
    )


def test_at_least_two_error_types_have_statistical_power():
    """A rule with n=3 cannot support any claim, however good its recall looks."""
    rows = _corpus()
    counts = Counter(r["ground_truth"]["rule"] for r in _injected(rows))
    with_power = [rule for rule, n in counts.items() if n >= MIN_ITEMS_FOR_POWER]
    assert len(with_power) >= MIN_RULES_WITH_POWER, (
        f"only {with_power} reach n>={MIN_ITEMS_FOR_POWER}; the rest cannot be reported"
    )


def test_both_halves_of_a_pair_ship_identical_state():
    """If the state differed between halves it would be the answer key, not context."""
    rows = _corpus()
    by_pair = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(row)
    for pair_id, group in by_pair.items():
        states = {tuple(r.get("state_facts") or ()) for r in group}
        assert len(states) == 1, f"{pair_id} halves ship different state: {states}"


def test_state_scope_items_actually_carry_state():
    rows = _corpus()
    for row in rows:
        if row["context_scope"] == "state":
            assert row.get("state_facts"), f"{row['passage_id']} is state-scope but ships none"


def test_the_corpus_still_validates():
    rows = _corpus()
    assert bench.validate_corpus(rows) == []


def test_a_constant_strategy_can_never_look_competent():
    """Flag-everything must score J 0. If this moves, the headline metric is broken."""
    evaluate = _evaluate()

    items = []
    for n in range(20):
        items += [
            {"pair_id": f"p{n}", "novel_id": "x", "text": "a\n\nb",
             "injected_location": {"paragraph_index": 0},
             "ground_truth": {"has_error": True, "rule": "r"}},
            {"pair_id": f"p{n}", "novel_id": "x", "text": "a\n\nb",
             "injected_location": None,
             "ground_truth": {"has_error": False, "rule": None}},
        ]
    always = evaluate.score(items, lambda i: evaluate.Prediction(has_error=True))
    never = evaluate.score(items, lambda i: evaluate.Prediction(has_error=False))
    assert always.youden_j == 0.0
    assert never.youden_j == 0.0
    assert round(always.f1, 3) == 0.667, "the flag-everything F1 floor moved"


def test_a_perfect_predictor_still_scores_one():
    """The other end of the scale: if this breaks, nothing can ever look good."""
    evaluate = _evaluate()

    items = [
        {"pair_id": "p1", "novel_id": "x", "text": "a\n\nb",
         "injected_location": {"paragraph_index": 1},
         "ground_truth": {"has_error": True, "rule": "r"}},
        {"pair_id": "p1", "novel_id": "x", "text": "a\n\nb",
         "injected_location": None,
         "ground_truth": {"has_error": False, "rule": None}},
    ]
    scores = evaluate.score(
        items,
        lambda i: evaluate.Prediction(
            has_error=i["ground_truth"]["has_error"], paragraph_index=1
        ),
    )
    assert scores.youden_j == 1.0
    assert scores.localization == 1.0


def test_confidence_still_produces_an_auc():
    """AUC is what separates 'cannot separate' from 'thresholded badly'."""
    evaluate = _evaluate()

    items = []
    for n in range(10):
        items += [
            {"pair_id": f"p{n}", "novel_id": "x", "text": "a",
             "injected_location": {"paragraph_index": 0},
             "ground_truth": {"has_error": True, "rule": "r"}},
            {"pair_id": f"p{n}", "novel_id": "x", "text": "a", "injected_location": None,
             "ground_truth": {"has_error": False, "rule": None}},
        ]
    scored = evaluate.score(
        items,
        lambda i: evaluate.Prediction(
            has_error=True, confidence=0.9 if i["ground_truth"]["has_error"] else 0.1
        ),
    )
    assert scored.roc_auc == 1.0


def test_every_predictor_still_declares_a_kind_and_key_need():
    """A model landing in the heuristic suite made `attack` a 58-minute run once."""
    evaluate = _evaluate()

    for name, predict in evaluate.PREDICTORS.items():
        assert predict.kind in ("model", "attack", "floor", "diagnostic", "human"), name
        assert isinstance(predict.needs_key, bool), name
        if predict.kind in evaluate.HEURISTIC_KINDS:
            assert predict.needs_key is False, name


def test_state_anchored_items_are_unsolvable_without_their_state():
    """The rule's premise: the contradicted fact is not in the passage.

    If the anchor leaked into the text, these would be ordinary passage items and the
    state would be decoration.
    """
    rows = _corpus()
    checked = 0
    for row in _injected(rows):
        if row["ground_truth"]["rule"] != "state_trait_flip":
            continue
        checked += 1
        # assert exactly what the generator enforces: the (owner, noun, value) triple
        # is unique. Rejecting any surviving phrase that merely shares the colour word
        # would fail on a valid item, since black/brown/grey are shared between eye and
        # hair - "Anna's black hair" flipped beside "his black eyes" is legitimate
        owner, anchor_noun, anchor_value = bench.parse_trait_fact(row["state_facts"][0])
        for noun, pattern in bench.TRAIT_PATTERNS:
            if noun != anchor_noun:
                continue
            for m in pattern.finditer(row["text"]):
                stated = (m.group("owner").lower(), m.group("value").lower())
                assert stated != (owner.lower(), anchor_value), (
                    f"{row['passage_id']} still states {m.group(0)!r}, the very trait its "
                    "state anchors; that is a trait_flip item, not state-anchored"
                )
    assert checked > 0, "no state_trait_flip items to check"


if __name__ == "__main__":
    raise SystemExit(1 if _selftest.run(vars().copy()) else 0)
