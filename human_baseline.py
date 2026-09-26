#!/usr/bin/env python3
"""Measure how well a careful human reader does on this benchmark.

Every model tested so far scores J near zero, and there are two very different
explanations: the models are bad, or the items are not solvable by a careful reader.
Nothing else on the leaderboard distinguishes those, and publishing a benchmark
without knowing which is true is the real risk. This is the instrument that settles it.

Methodology, because a sloppy human baseline is worse than none:

- One half of each matched pair, never both. The two halves differ in a single token,
  so a reader shown both is running the pair_diff_targeted attack by hand. Reuses
  bench.one_half_per_pair, the same guard the public release uses.
- No feedback during the run. Being told you were right teaches the injection pattern,
  and then you are measuring how fast someone learns the generator, not how well a
  reader spots continuity errors.
- Answers save after every item, so a run can be stopped and resumed. A tired reader
  guessing to reach the end is a worse baseline than a short honest one.
- Time per item is recorded. A human who needs four minutes a passage is a different
  proposition from one who needs twenty seconds, and the pipeline is meant to replace
  the expensive version.

    python3 human_baseline.py --n 40          # answer 40 passages
    python3 human_baseline.py --score         # score what has been answered so far
    python3 human_baseline.py --review        # show the answers with ground truth
"""

from __future__ import annotations

import argparse
import json
import sys
import random
import statistics
import textwrap
import time
from pathlib import Path

import bench

# evaluate is imported lazily, not here: it pulls in pydantic and the provider SDKs,
# and the one script in this project a person runs by hand should start with a plain
# `python3` in any directory. Only --score needs it.

ROOT = Path(__file__).resolve().parent
DEFAULT_ANSWERS = ROOT / "human_answers.json"
PROMPT = "continuity error in this passage?  [y]es  [n]o  [s]kip  [q]uit > "


def answered_items(items, answers):
    """Items with a real verdict, excluding skips."""
    return [
        item for item in items
        if item["passage_id"] in answers and not answers[item["passage_id"]].get("skipped")
    ]


def load_answers(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_answers(path, answers):
    Path(path).write_text(json.dumps(answers, indent=2), encoding="utf-8")


def candidate_items(corpus, split, seed):
    """One half of each matched pair, shuffled, so neither twin reveals the other."""
    rows = [json.loads(line) for line in Path(corpus).open(encoding="utf-8")]
    items = bench.one_half_per_pair([r for r in rows if r["split"] == split], seed)
    random.Random(seed).shuffle(items)
    return items


def render(item, index, total):
    paragraphs = item["text"].split("\n\n")
    width = 96
    lines = [
        "=" * width,
        f"passage {index}/{total}   ({len(paragraphs)} paragraphs, {len(item['text'])} chars)",
        "=" * width,
        "",
    ]
    for n, paragraph in enumerate(paragraphs):
        lines.append(f"[{n}]")
        lines.extend(textwrap.wrap(paragraph, width=width, initial_indent="    ",
                                   subsequent_indent="    ") or ["    "])
        lines.append("")
    return "\n".join(lines)


def ask(item, index, total):
    """Show one passage and collect a verdict. Returns None if the reader quit."""
    print("\n" * 3 + render(item, index, total))
    started = time.perf_counter()
    while True:
        reply = input(PROMPT).strip().lower()
        if reply in ("q", "quit"):
            return None
        if reply in ("s", "skip"):
            return {"skipped": True, "seconds": round(time.perf_counter() - started, 1)}
        if reply in ("n", "no"):
            return {"has_error": False, "paragraph_index": None,
                    "seconds": round(time.perf_counter() - started, 1)}
        if reply in ("y", "yes"):
            count = len(item["text"].split("\n\n"))
            while True:
                which = input(f"  which paragraph? [0-{count - 1}, or ? if unsure] > ").strip()
                if which == "?":
                    paragraph = None
                    break
                if which.isdigit() and 0 <= int(which) < count:
                    paragraph = int(which)
                    break
                print("  not a paragraph number")
            return {"has_error": True, "paragraph_index": paragraph,
                    "seconds": round(time.perf_counter() - started, 1)}
        print("  answer y, n, s or q")


def score_answers(items, answers):
    """Score the answered subset through the shared harness, as any predictor is."""
    import evaluate

    answered = answered_items(items, answers)
    if not answered:
        return None, None

    def predict(item):
        reply = answers[item["passage_id"]]
        return evaluate.Prediction(
            has_error=bool(reply["has_error"]),
            paragraph_index=reply.get("paragraph_index"),
            note=f"{reply['seconds']}s",
        )

    return answered, evaluate.score(answered, predict)


def cmd_review(items, answers):
    wrong = 0
    for item in answered_items([i for i in items], answers):
        passage_id = item["passage_id"]
        reply = answers[passage_id]
        truth = item["ground_truth"]["has_error"]
        correct = bool(reply["has_error"]) == truth
        wrong += not correct
        mark = "ok  " if correct else "MISS"
        detail = f"said {'error' if reply['has_error'] else 'clean'}, truth {'error' if truth else 'clean'}"
        extra = ""
        if truth and reply["has_error"]:
            said, actually = reply.get("paragraph_index"), item["injected_location"]["paragraph_index"]
            extra = f"  paragraph said {said} actual {actually}"
            if said != actually:
                extra += "  (located wrong)"
        print(f"  {mark} {passage_id:26} {detail}{extra}")
        if not correct and truth:
            print(f"       injected: {item['ground_truth']['description']}")
    print(f"\n{wrong} wrong out of {len(answered_items(items, answers))}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default=str(ROOT / "corpus" / "continuity_v0.jsonl"))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--answers", default=str(DEFAULT_ANSWERS))
    parser.add_argument("--seed", default="human-baseline-v0")
    # both spellings, because the docs said --n and argparse only offered -n
    parser.add_argument("-n", "--n", "--count", dest="n", type=int, default=40,
                        help="passages to answer this sitting")
    parser.add_argument("--score", action="store_true", help="score answers and stop")
    parser.add_argument("--review", action="store_true", help="show answers against truth")
    args = parser.parse_args(argv)

    items = candidate_items(args.corpus, args.split, args.seed)
    answers = load_answers(args.answers)

    if args.review:
        return cmd_review(items, answers)

    if not args.score and not sys.stdin.isatty():
        # input() on a pipe hits EOF immediately and every passage would record as a
        # skip, quietly producing an empty baseline that looks like a completed one
        print(
            "This needs a real terminal: stdin is not a TTY, so the prompts cannot be\n"
            "answered and every passage would be skipped.\n\n"
            "Open your own terminal window and run:\n"
            f"    cd {ROOT}\n"
            f"    python3 human_baseline.py --n {args.n}\n\n"
            "--score and --review work fine from anywhere."
        )
        return 2

    if not args.score:
        todo = [i for i in items if i["passage_id"] not in answers][: args.n]
        if not todo:
            print("nothing left unanswered; use --score or raise -n")
        for offset, item in enumerate(todo, 1):
            reply = ask(item, offset, len(todo))
            if reply is None:
                print("\nstopped. progress saved.")
                break
            answers[item["passage_id"]] = reply
            save_answers(args.answers, answers)
        print(f"\n{len(answers)} answered in total, saved to {args.answers}")

    answered, scores = score_answers(items, answers)
    if scores is None:
        print("no scored answers yet")
        return 0
    import evaluate

    median = statistics.median(answers[i["passage_id"]]["seconds"] for i in answered)
    # kind="human", not "model". A human published as a model would render at $0.00 on
    # a cost-vs-quality chart - making the most expensive predictor in existence the
    # cheapest point on it - and would publish a latency of ~0.0001s, because score()
    # times a dict lookup here rather than the reading. The honest cost axis for a
    # human is seconds per passage, which is what this records.
    path, payload = evaluate.write_results(
        "human", args.corpus, answered, scores,
        extra={"median_seconds_per_passage": median, "answered": len(answered),
               "items_note": "one half of each matched pair, so not the model item pool"},
        kind="human",
    )
    evaluate.print_row("human", payload)
    print(f"median {median}s per passage over {len(answered)} answered")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
