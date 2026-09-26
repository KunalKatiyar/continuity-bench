"""Self-checks for nli.py. Pure functions only, no model download, no network."""

import nli


def test_all_ordered_pairs_are_generated():
    pairs = nli.paragraph_pairs("a\n\nb\n\nc\n\nd")
    assert len(pairs) == 6
    assert ("a", "d", 3) in pairs
    assert ("c", "d", 3) in pairs
    assert all(premise != hypothesis for premise, hypothesis, _ in pairs)


def test_the_pair_index_points_at_the_later_paragraph():
    """Localization is the paragraph that contradicts, not the one it contradicts."""
    for premise, hypothesis, index in nli.paragraph_pairs("p0\n\np1\n\np2"):
        assert hypothesis == f"p{index}"


def test_a_single_paragraph_yields_no_pairs():
    assert nli.paragraph_pairs("only one paragraph") == []
    assert nli.paragraph_pairs("") == []


def test_blank_paragraphs_are_dropped():
    assert len(nli.paragraph_pairs("a\n\n\n\nb")) == 1


def test_long_passages_are_capped():
    text = "\n\n".join(f"p{i}" for i in range(40))
    pairs = nli.paragraph_pairs(text, max_paragraphs=10)
    assert max(index for _, _, index in pairs) == 9
    assert len(pairs) == 45


def test_auc_is_one_for_perfect_separation_and_half_for_none():
    assert nli.roc_auc([(0.9, True), (0.8, True), (0.2, False), (0.1, False)]) == 1.0
    assert nli.roc_auc([(0.1, True), (0.2, True), (0.8, False), (0.9, False)]) == 0.0
    assert nli.roc_auc([(0.5, True), (0.5, False)]) == 0.5


def test_auc_handles_ties_without_crediting_them():
    """All-equal scores separate nothing, whatever the class balance."""
    assert nli.roc_auc([(0.4, True)] * 3 + [(0.4, False)] * 3) == 0.5


def test_auc_is_threshold_free():
    """Shifting every score by a constant cannot change separation."""
    base = [(0.6, True), (0.55, True), (0.5, False), (0.1, False)]
    shifted = [(p - 0.05, truth) for p, truth in base]
    assert nli.roc_auc(base) == nli.roc_auc(shifted)


def test_auc_with_one_class_is_undefined_and_returns_zero():
    assert nli.roc_auc([(0.9, True), (0.8, True)]) == 0.0
    assert nli.roc_auc([]) == 0.0


def test_threshold_sweep_finds_the_separating_cut():
    best, curve = nli.sweep_thresholds([(0.9, True), (0.8, True), (0.2, False), (0.1, False)])
    assert best["youden_j"] == 1.0
    assert 0.2 < best["threshold"] <= 0.8
    assert len(curve) == 4  # one candidate cut per observed score, not a fixed grid


def test_the_sweep_can_reach_saturated_scores():
    """A fixed 0.01 grid tops out at 0.99 and cannot separate scores above it.

    MNLI contradiction softmax saturates: on the real corpus the median is 0.986 and
    45% of items sit above 0.99, so a grid would report a cut that is an artefact of
    where the grid stopped.
    """
    scored = [(0.9999, True), (0.9995, True), (0.9990, False), (0.9985, False)]
    best, _ = nli.sweep_thresholds(scored)
    assert best["youden_j"] == 1.0
    assert best["threshold"] > 0.99


def test_threshold_sweep_reports_zero_when_nothing_separates():
    best, _ = nli.sweep_thresholds([(0.5, True), (0.5, False)] * 10)
    assert best["youden_j"] == 0.0


def test_threshold_sweep_recall_and_fpr_are_consistent_with_j():
    _, curve = nli.sweep_thresholds([(0.9, True), (0.3, True), (0.6, False), (0.1, False)])
    for row in curve:
        assert abs(row["youden_j"] - (row["recall"] - row["fpr"])) < 1e-9


if __name__ == "__main__":
    import _selftest

    raise SystemExit(1 if _selftest.run(vars().copy()) else 0)
