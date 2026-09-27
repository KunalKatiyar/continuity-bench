"""Self-checks for build_site.py. Run directly (python test_build_site.py) or under pytest."""

import json
import tempfile
from pathlib import Path

import build_site


def _row(name, j=0.5, f1=0.6, cost=0.0, kind=None, **over):
    row = {
        "predictor": name,
        "corpus": "c.jsonl",
        "items": 100,
        "pairs": 50,
        "novels": 5,
        "youden_j": j,
        "f1": f1,
        "precision": 0.5,
        "recall": 0.6,
        "false_positive_rate": 0.1,
        "localization": 0.3,
        "cost_usd": cost,
        "input_tokens": 0,
        "output_tokens": 0,
        "median_latency_s": 0.0,
        "recall_by_novel": {},
    }
    row.update(over)
    if kind is not None:
        row["kind"] = kind
    row["kind"] = build_site.kind_of(row)
    return row


def test_kind_comes_from_the_declared_field_not_the_name():
    assert build_site.kind_of({"predictor": "jev_naive", "kind": "model"}) == "model"
    assert build_site.kind_of({"predictor": "pair_leak", "kind": "diagnostic"}) == "diagnostic"


def test_kind_falls_back_to_a_name_guess_for_older_results():
    assert build_site.kind_of({"predictor": "always_clean"}) == "floor"
    assert build_site.kind_of({"predictor": "llm_claude-opus-5_high"}) == "model"
    assert build_site.kind_of({"predictor": "corpus_prior"}) == "attack"


def test_a_future_baseline_is_not_mislabelled_a_free_heuristic():
    row = _row("jev_naive", kind="model", cost=0.001)
    assert row["kind"] == "model"
    assert "Free heuristic" not in build_site.legend([row])


def test_a_single_paid_run_is_enough_for_a_cost_axis():
    assert build_site.has_paid_run([_row("a"), _row("b")]) is False
    assert build_site.has_paid_run([_row("llm_x", cost=0.004)]) is True
    assert build_site.has_paid_run([_row("a", cost=0.0), _row("llm_x", cost=0.004)]) is True


def test_bars_are_used_when_every_run_is_free():
    html = build_site.render([_row("a"), _row("b", j=0.1)], "stats")
    assert "class=\"bar\"" in html
    assert "cost per passage" not in html
    assert "no cost axis to draw yet" in html


def test_scatter_is_used_once_a_run_costs_money():
    html = build_site.render([_row("a", cost=0.0), _row("llm_x_high", cost=0.004)], "stats")
    assert "cost per passage" in html
    assert "no cost axis to draw yet" not in html


def test_the_status_line_cannot_contradict_the_charts():
    free = build_site.render([_row("corpus_prior", j=0.066)], "stats")
    assert "No model has been run" in free
    paid = build_site.render(
        [_row("corpus_prior", j=0.066), _row("llm_x_high", kind="model", cost=0.004, j=0.55)],
        "stats",
    )
    assert "No model has been run" not in paid
    assert "llm_x_high" in paid


def test_the_status_line_derives_the_attack_figure_from_the_rows():
    html = build_site.render([_row("corpus_prior", j=0.123)], "stats")
    assert "+0.123" in html


def test_legend_only_lists_kinds_present():
    legend = build_site.legend([_row("corpus_prior")])
    assert "Free heuristic" in legend
    assert "Model" not in legend
    assert "Trivial floor" not in legend


def test_results_are_sorted_by_descending_discrimination():
    with tempfile.TemporaryDirectory() as tmp:
        for name, j in (("a", 0.1), ("b", 0.9), ("c", 0.5)):
            Path(tmp, f"{name}.json").write_text(json.dumps(_row(name, j=j)), encoding="utf-8")
        rows = build_site.load_results(tmp)
    assert [r["predictor"] for r in rows] == ["b", "c", "a"]


def test_results_without_the_current_metrics_are_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "old.json").write_text(json.dumps({"predictor": "old", "f1": 0.5}), encoding="utf-8")
        Path(tmp, "new.json").write_text(json.dumps(_row("new")), encoding="utf-8")
        rows = build_site.load_results(tmp)
    assert [r["predictor"] for r in rows] == ["new"]


def test_markup_in_a_predictor_name_is_escaped():
    html = build_site.render([_row("<script>x</script>"), _row("b", j=0.2)], "stats")
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html


def test_dark_mode_redefines_the_surface_and_the_series_step():
    html = build_site.render([_row("a")], "stats")
    assert 'prefers-color-scheme: dark' in html
    assert ':root[data-theme="dark"]' in html
    assert "#d95926" in html
    assert "background: var(--plane)" in html


def test_an_empty_results_dir_renders_a_page_not_a_crash():
    html = build_site.render([], "")
    assert "No results yet" in html
    assert "<table" not in html


if __name__ == "__main__":
    import _selftest

    raise SystemExit(1 if _selftest.run(vars().copy()) else 0)
