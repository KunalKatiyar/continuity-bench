#!/usr/bin/env python3
"""Generate the static leaderboard from results/*.json.

Writes site/index.html - one self-contained file, no build step and no JavaScript
libraries, so it can be served straight from GitHub Pages.

The headline chart is cost against F1, as the project brief asks for, with the
"flag every passage" F1 of 0.667 drawn as a reference line: on a balanced paired
corpus that strategy discriminates nothing, so any point at or below that line is
worthless however good its F1 looks. A second chart plots the same runs against
J = recall - FPR, which has an honest zero.

ponytail: the SVG is emitted as f-strings rather than drawn with a plotting library,
because two scatter panels and a reference line do not justify a dependency or a
build step. Reach for a real library when this needs axes it cannot fake.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
SITE_DIR = ROOT / "site"

FLOOR_F1 = 2 / 3

KINDS = {
    "model": ("Model", "#2a78d6", "#3987e5"),
    "attack": ("Free heuristic", "#eb6834", "#d95926"),
    "diagnostic": ("Integrity check", "#1baf7a", "#199e70"),
    "floor": ("Trivial floor", "#898781", "#898781"),
}

DIAGNOSTIC_NAMES = {"pair_leak", "pair_diff_targeted"}

FLOOR_PREDICTORS = {"always_clean", "always_error"}


def kind_of(row):
    """The predictor's declared kind, falling back to a name guess for older files.

    evaluate.py persists `kind`, so the chart no longer classifies by name prefix -
    that would publish a future `jev_naive` or `qwen3_local` baseline in the legend
    as a free heuristic, which is a wrong claim rather than a crash.
    """
    kind = row.get("kind")
    if kind in KINDS:
        return kind
    name = row["predictor"]
    if name in FLOOR_PREDICTORS:
        return "floor"
    if name in DIAGNOSTIC_NAMES:
        return "diagnostic"
    return "model" if name.startswith("llm") else "attack"


def load_results(results_dir):
    """Read every results JSON, newest schema only, sorted by descending J."""
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if "youden_j" not in row:
            continue
        row["kind"] = kind_of(row)
        rows.append(row)
    rows.sort(key=lambda r: (-r["youden_j"], r["predictor"]))
    return rows


def has_paid_run(rows):
    """True once some run cost money, which is what gives a cost axis anything to show.

    Testing for two distinct cost values instead would draw bars, and print "every run
    so far is free", beside a lone paid run that cost real money.
    """
    return any(row["cost_usd"] > 0 for row in rows)


def scatter(rows, y_key, y_label, floor=None, floor_label=None):
    """Inline SVG scatter of cost against one quality metric.

    Only used once runs actually differ in cost - see has_cost_spread. Marks are
    circles with a 2px surface ring so overlapping runs stay readable, each carries
    a <title> for the hover tooltip, and labels sit beside their own point rather
    than being pushed away to avoid collisions: a label that has drifted onto a
    different y value is worse than no label, so a colliding one is dropped and the
    hover tooltip and table carry it instead.
    """
    width, height = 560, 330
    pad_l, pad_r, pad_t, pad_b = 62, 112, 22, 48
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    x_max = max(row["cost_usd"] for row in rows) or 1.0
    y_values = [row[y_key] for row in rows]
    y_lo, y_hi = min(0.0, min(y_values)), max(1.0, max(y_values))

    def sx(cost):
        return pad_l + (cost / x_max) * plot_w

    def sy(value):
        return pad_t + plot_h - (value - y_lo) / (y_hi - y_lo) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Cost against {y_label} for each run" class="chart">'
    ]
    for i in range(6):
        value = y_lo + (y_hi - y_lo) * i / 5
        y = sy(value)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" class="grid"/>'
            f'<text x="{pad_l - 10}" y="{y + 4:.1f}" class="tick tick-y">{value:.2f}</text>'
        )
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" '
        f'y2="{pad_t + plot_h}" class="axis"/>'
    )
    for i in range(3):
        cost = x_max * i / 2
        parts.append(
            f'<text x="{sx(cost):.1f}" y="{pad_t + plot_h + 20:.1f}" '
            f'class="tick tick-x">${cost:.4f}</text>'
        )
    parts.append(
        f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 8}" class="axis-title">'
        f"cost per passage (USD)</text>"
        f'<text x="15" y="{pad_t + plot_h / 2:.1f}" class="axis-title" '
        f'transform="rotate(-90 15 {pad_t + plot_h / 2:.1f})">{y_label}</text>'
    )
    if floor is not None:
        y = sy(floor)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" class="floor"/>'
            f'<text x="{pad_l + plot_w - 2:.1f}" y="{y - 7:.1f}" class="floor-label" '
            f'text-anchor="end">{esc(floor_label)}</text>'
        )
    placed = []
    for row in rows:
        x, y = sx(row["cost_usd"]), sy(row[y_key])
        _, light, dark = KINDS[row["kind"]]
        parts.append(
            f'<g class="mark"><title>{esc(tooltip(row, y_key, y_label))}</title>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" class="ring"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" '
            f'style="--c:{light};--cd:{dark}" class="dot"/></g>'
        )
        if not any(abs(y - py) < 12 and abs(x - px) < 80 for px, py in placed):
            placed.append((x, y))
            parts.append(
                f'<text x="{x + 10:.1f}" y="{y + 4:.1f}" class="point-label">'
                f'{esc(row["predictor"])}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def bars(rows, y_key, y_label, floor=None):
    """Inline SVG horizontal bars, for when every run costs the same.

    A cost scatter with one distinct cost value would draw an axis range that does
    not exist in the data, so the form follows the data: comparing a metric across
    named runs is a bar chart, and the cost axis appears only once a paid run gives
    it something to show.
    """
    width, row_h = 560, 30
    pad_l, pad_r, pad_t, pad_b = 132, 58, 14, 40
    plot_w = width - pad_l - pad_r
    height = pad_t + pad_b + row_h * len(rows)
    values = [row[y_key] for row in rows]
    lo, hi = min(0.0, min(values)), max(1.0, max(values))
    signed = lo < 0

    def sx(value):
        return pad_l + (value - lo) / (hi - lo) * plot_w

    zero = sx(0.0)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{y_label} for each run" class="chart">'
    ]
    for i in range(6):
        value = lo + (hi - lo) * i / 5
        x = sx(value)
        parts.append(
            f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t + row_h * len(rows)}" '
            f'class="grid"/>'
            f'<text x="{x:.1f}" y="{height - 22}" class="tick tick-x">{value:.2f}</text>'
        )
    if floor is not None:
        x = sx(floor)
        base = pad_t + row_h * len(rows)
        parts.append(
            f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{base}" class="floor"/>'
        )
    for index, row in enumerate(rows):
        _, light, dark = KINDS[row["kind"]]
        value = row[y_key]
        top = pad_t + index * row_h + 6
        bar_h = row_h - 13
        x0, x1 = min(zero, sx(value)), max(zero, sx(value))
        parts.append(
            f'<text x="{pad_l - 12}" y="{top + bar_h / 2 + 4:.1f}" class="bar-label">'
            f'{esc(row["predictor"])}</text>'
            f'<g class="mark"><title>{esc(tooltip(row, y_key, y_label))}</title>'
            f'<rect x="{x0:.1f}" y="{top:.1f}" width="{max(1.5, x1 - x0):.1f}" '
            f'height="{bar_h}" rx="3" style="--c:{light};--cd:{dark}" class="bar"/></g>'
            f'<text x="{x1 + 7:.1f}" y="{top + bar_h / 2 + 4:.1f}" class="bar-value">'
            + (f"{value:+.3f}" if signed else f"{value:.3f}")
            + "</text>"
        )
    parts.append(
        f'<line x1="{zero:.1f}" y1="{pad_t}" x2="{zero:.1f}" '
        f'y2="{pad_t + row_h * len(rows)}" class="axis"/>'
        f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 7}" class="axis-title">'
        f"{esc(y_label)}</text></svg>"
    )
    return "".join(parts)


def ci95(row):
    """The J interval as text, flagged when it spans zero and so proves nothing."""
    interval = row.get("youden_j_ci95")
    if not interval:
        return "&mdash;"
    lo, hi = interval
    spans_zero = not row.get("youden_j_significant", lo > 0 or hi < 0)
    text = f"{lo:+.3f} to {hi:+.3f}"
    return f"{text} &#8225;" if spans_zero else text


def per_passage(row):
    """Cost of one passage for this run."""
    return row["cost_usd"] / row["items"] if row["items"] else 0.0


def tooltip(row, y_key, y_label):
    return (
        f'{row["predictor"]} - {y_label} {row[y_key]:.3f}, recall {row["recall"]:.3f}, '
        f'FPR {row["false_positive_rate"]:.3f}, localization {row["localization"]:.3f}, '
        f"${per_passage(row):.5f} per passage"
    )


def esc(text):
    """Escape text for an SVG or HTML text node or attribute."""
    return html.escape(str(text))


def legend(rows):
    """Legend for the kinds actually present - never advertise an empty category."""
    present = {row["kind"] for row in rows}
    items = "".join(
        f'<span class="legend-item"><span class="swatch" '
        f'style="--c:{light};--cd:{dark}"></span>{label}</span>'
        for kind, (label, light, dark) in KINDS.items()
        if kind in present
    )
    return f'<div class="legend">{items}</div>'


def table(rows):
    head = (
        "<thead><tr><th>predictor</th><th>J</th><th>J 95% CI</th><th>F1</th>"
        "<th>precision</th><th>recall</th><th>FPR</th><th>localization</th>"
        "<th>$/passage</th><th>items</th></tr></thead>"
    )
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f'<td class="name">{esc(row["predictor"])}</td>'
            f'<td class="num strong">{row["youden_j"]:+.3f}</td>'
            + f'<td class="num ci">{ci95(row)}</td>'
            f'<td class="num">{row["f1"]:.3f}</td>'
            f'<td class="num">{row["precision"]:.3f}</td>'
            f'<td class="num">{row["recall"]:.3f}</td>'
            f'<td class="num">{row["false_positive_rate"]:.3f}</td>'
            f'<td class="num">{row["localization"]:.3f}</td>'
            f'<td class="num">${per_passage(row):.5f}</td>'
            f'<td class="num">{row["items"]}</td>'
            "</tr>"
        )
    return f'<table>{head}<tbody>{"".join(body)}</tbody></table>'


STYLE = """
:root {
  color-scheme: light dark;
  --surface: #fcfcfb; --plane: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --mode: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --surface: #1a1a19; --plane: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835; --mode: dark;
  }
  :root:not([data-theme="light"]) .dot, :root:not([data-theme="light"]) .bar,
  :root:not([data-theme="light"]) .swatch { --c: var(--cd); }
}
:root[data-theme="dark"] {
  --surface: #1a1a19; --plane: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --axis: #383835;
}
:root[data-theme="dark"] .dot, :root[data-theme="dark"] .bar,
:root[data-theme="dark"] .swatch { --c: var(--cd); }
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--plane); color: var(--ink);
  font: 15px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 1120px; margin: 0 auto; padding: 48px 16px 72px; }
h1 { font-size: 1.75rem; letter-spacing: -0.02em; margin: 0 0 6px; }
h2 { font-size: 1.05rem; letter-spacing: -0.01em; margin: 40px 0 4px; }
p { color: var(--ink-2); margin: 0 0 8px; max-width: 70ch; }
.sub { color: var(--muted); font-size: 0.875rem; }
.panels { display: flex; flex-wrap: wrap; gap: 20px; margin-top: 12px; }
.panel {
  background: var(--surface); border: 1px solid var(--grid); border-radius: 10px;
  padding: 14px 10px 6px; flex: 1 1 380px; min-width: 0;
}
.panel h3 { margin: 0 0 2px 10px; font-size: 0.9rem; font-weight: 600; }
.panel .note { margin: 0 0 4px 10px; font-size: 0.8rem; color: var(--muted); }
.chart { width: 100%; height: auto; display: block; overflow: visible; }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.floor { stroke: var(--muted); stroke-width: 1.5; stroke-dasharray: 5 4; }
.floor-label, .tick { fill: var(--muted); font-size: 10.5px; }
.tick-y { text-anchor: end; }
.tick-x { text-anchor: middle; }
.axis-title { fill: var(--ink-2); font-size: 11.5px; text-anchor: middle; }
.point-label { fill: var(--ink-2); font-size: 11px; }
.bar-label { fill: var(--ink-2); font-size: 11.5px; text-anchor: end;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.bar-value { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.bar { fill: var(--c); }
/* the aqua slot sits below 3:1 on the light surface, so every mark keeps a visible
   direct label and the full table below acts as the relief the palette requires */
.mark:hover .bar { opacity: 0.82; }
.dot { fill: var(--c); }
.ring { fill: var(--surface); }
.mark { cursor: help; }
.mark:hover .dot { r: 6.5; }
.legend { display: flex; gap: 18px; flex-wrap: wrap; margin: 16px 0 0; font-size: 0.85rem; color: var(--ink-2); }
.legend-item { display: inline-flex; align-items: center; gap: 7px; }
.swatch { width: 11px; height: 11px; border-radius: 3px; background: var(--c); }
table { border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 0.85rem; }
th, td { padding: 7px 10px; border-bottom: 1px solid var(--grid); text-align: left; }
th { color: var(--muted); font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.strong { font-weight: 600; color: var(--ink); }
td.ci { color: var(--muted); font-size: 0.78rem; white-space: nowrap; }
td.name { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem; }
.empty { background: var(--surface); border: 1px dashed var(--axis); border-radius: 10px; padding: 24px; color: var(--muted); }
@media (max-width: 640px) { main { padding: 28px 16px 48px; } .panel { flex: 1 1 100%; } }
"""


def headline(rows):
    """One line of status derived from the rows, so the page cannot contradict itself."""
    if not rows:
        return ""
    attacks = [row["youden_j"] for row in rows if row["kind"] == "attack"]
    diagnostics = [row for row in rows if row["kind"] == "diagnostic"]
    models = [row for row in rows if row["kind"] == "model"]
    parts = []
    if attacks:
        parts.append(
            f"The best free heuristic attack reaches J {max(attacks):+.3f}, so the corpus "
            "does not leak its own injection method."
        )
    if diagnostics:
        worst = max(diagnostics, key=lambda row: row["youden_j"])
        parts.append(
            f'Integrity checks are not baselines: {worst["predictor"]} reaches '
            f'J {worst["youden_j"]:+.3f} by exploiting the matched-pair design, which is '
            "why the released test split ships one half of each pair."
        )
    if not models:
        parts.append("No model has been run against it yet.")
    else:
        best = max(models, key=lambda row: row["youden_j"])
        parts.append(
            f'Best model so far: {best["predictor"]} at J {best["youden_j"]:+.3f}, '
            f"${per_passage(best):.5f} per passage."
        )
    return f'<p class="sub">{esc(" ".join(parts))}</p>'


def render(rows, corpus_stats):
    if not rows:
        body = '<div class="empty">No results yet. Run <code>python3 evaluate.py attack</code>.</div>'
    elif has_paid_run(rows):
        body = (
            '<div class="panels">'
            '<div class="panel"><h3>Cost vs F1</h3>'
            '<p class="note">Dashed line: flagging every passage. Points on or below it '
            "discriminate nothing, whatever their F1.</p>"
            + scatter(rows, "f1", "F1", FLOOR_F1, "flag-everything F1 = 0.667")
            + "</div>"
            '<div class="panel"><h3>Cost vs discrimination</h3>'
            '<p class="note">J = recall &minus; false-positive rate. Zero means no '
            "discrimination; 1 is perfect.</p>"
            + scatter(rows, "youden_j", "J = recall - FPR", 0.0, "no discrimination")
            + "</div></div>"
        )
    else:
        body = (
            '<div class="panels">'
            '<div class="panel"><h3>Discrimination</h3>'
            '<p class="note">J = recall &minus; false-positive rate. Zero means the run '
            "separates nothing.</p>"
            + bars(rows, "youden_j", "J = recall - FPR")
            + "</div>"
            '<div class="panel"><h3>F1, for comparison</h3>'
            '<p class="note">Dashed line: flagging every passage scores 0.667 here while '
            "discriminating nothing. This is why F1 is not the headline.</p>"
            + bars(rows, "f1", "F1", FLOOR_F1)
            + "</div></div>"
            '<p class="sub">Every run so far is free, so there is no cost axis to draw yet. '
            "The cost-vs-quality scatter this benchmark is built around appears once a paid "
            "model run lands.</p>"
        )
    if rows:
        body += (
            legend(rows)
            + "<h2>All runs</h2>"
            + f'<p class="sub">{esc(corpus_stats)}</p>'
            + table(rows)
            + '<p class="sub">&#8225; the 95% interval spans zero, so that run does not '
            "establish any discrimination either way &mdash; usually too few items.</p>"
        )
    status = headline(rows)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Continuity Bench</title>
<style>{STYLE}</style>
</head>
<body>
<main>
<h1>Continuity Bench</h1>
<p>Detecting injected continuity errors in novel-length fiction. Every injected
passage ships with the same passage unedited as its control, so the false-positive
rate is measured on the same text distribution as the recall.</p>
{status}
{body}
</main>
</body>
</html>
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(RESULTS_DIR))
    parser.add_argument("--out", default=str(SITE_DIR / "index.html"))
    args = parser.parse_args(argv)
    rows = load_results(args.results)
    stats = ""
    if rows:
        stats = (
            f'{rows[0]["items"]} items from {rows[0]["novels"]} novels, '
            f'corpus {rows[0]["corpus"]}'
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows, stats), encoding="utf-8")
    print(f"wrote {out} ({len(rows)} run(s), {out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
