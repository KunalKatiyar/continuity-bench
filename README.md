# continuity-bench

A benchmark for detecting continuity errors in novel-length fiction, and (later) a
Jev-powered editing pipeline that tries to match full-context LLM accuracy at a
fraction of the cost.

**Status: Phase 1, step 1 of the build order.** The corpus builder works end to end.
No model has been run against it yet.

## What exists

`bench.py` — stdlib-only, no dependencies:

```bash
python3 bench.py fetch              # download the novels in books.tsv from Project Gutenberg
python3 bench.py build              # inject labeled errors, write corpus/continuity_v0.jsonl
python3 bench.py validate           # check every label, span and pair in a built corpus
python3 bench.py stats              # label/split/scope distribution
python3 bench.py sample --rule character_swap -n 5   # eyeball items with the edit marked
python3 test_bench.py               # 36 self-checks (also runs under pytest)
```

Current corpus: **298 matched pairs / 596 items from 28 novels**, every injected item
shipped with the same passage unedited as its control, so false-positive rate is
measured on exactly the same text distribution. Splits are assigned per novel, so no
novel appears in both `dev` and `test`.

## Item schema

```json
{
  "novel_id": "pg1260",
  "passage_id": "pg1260-p00456-err",
  "pair_id": "pg1260-p00456",
  "split": "dev",
  "context_scope": "passage",
  "n_paragraphs": 12,
  "text": "...the passage, with the error injected...",
  "error_type": "fact_contradiction",
  "injected_location": {
    "paragraph_index": 5, "char_start": 2317, "char_end": 2322,
    "original": "Bessie", "replacement": "Burns", "anchor_paragraph_index": 0
  },
  "ground_truth": {"has_error": true, "rule": "character_swap", "description": "..."}
}
```

The `-clean` twin of each item is byte-identical except for the injected span, with
`error_type: null` and `has_error: false`.

`context_scope` is `passage` when the contradicted fact is stated inside the same
passage, and `long_range` when it was established chapters earlier — the case that
cannot be solved by reading one passage, and the reason the Phase 2 story-state
pipeline exists.

## The injection rules, and what they actually yield

| rule | error type | items | what it does |
|---|---|---|---|
| `character_swap` | fact_contradiction | 292 | puts a character who is absent from the scene into it, in place of one who is present |
| `trait_flip` | fact_contradiction | 3 | flips the later of two matching eye/hair colour phrases with the same owner |
| `timeline_weekday` | timeline | 3 | breaks a day sequence the passage itself states ("Tuesday … the next morning … Wednesday") |

**The corpus is 98% one error type, and that is a finding, not a bug to paper over.**
Measured across all 35 downloaded novels (~70,000 paragraphs):

| pattern | occurrences | per novel |
|---|---|---|
| day-of-week mentions | 924 | 26.4 |
| …forming a checkable `DAY … connective … DAY` sequence | 3 | 0.1 |
| eye/hair colour phrases | 425 | 12.1 |
| …repeated with the same owner, so a flip is a contradiction | 32 | 0.9 |
| month + season co-occurrence | 25 | 0.7 (and mostly the modal verb "may", not May) |
| `Name's <kinship>` | 1 | 0.0 |

So: **rule-based injection over public-domain fiction supports exactly one
high-volume error type plus a long tail.** The facts that regex can verify
(a repeated trait phrase, an explicit weekday chain) are simply not repeated often
enough in real prose. Getting balanced volume across error types needs an LLM
injector that writes the contradiction rather than pattern-matching one. That is the
next step, and it is a measured conclusion rather than a guess.

## Known limitations

- **Training-data contamination.** All 35 novels are famous and certainly in every
  frontier model's training data. A model may "remember" that the eyes were blue
  rather than reason about the passage. Mitigations: report per-novel scores so
  memorisation shows up as variance, and swap in obscure Gutenberg texts for the
  release corpus. Not yet done.
- **Injected errors are not natural errors.** A programmatic edit may be detectable
  from a stylistic seam rather than from the continuity break. `character_swap` is
  the cleanest on this count (it substitutes a real name from the same novel) but the
  benchmark measures detection of *injected* errors, and the README should keep
  saying so.
- **No coreference.** `trait_flip` trusts the owner token ("her blue eyes") as the
  whole referent check. That holds inside a passage and fails across a novel, so
  pronoun-owned traits are restricted to `passage` scope and only possessive names
  ("Anna's hair") are allowed at long range. Two women in one passage both described
  as "her blue eyes" would still be mislabeled.
- **Name detection is heuristic.** A name qualifies as a person if it appears
  capitalised mid-sentence, speaks at least twice, and rarely follows an article.
  That was arrived at by fixing real failures: `The`, `Master`, `She`, `Company`,
  `Mars` and `Mrs` all got injected as characters before those filters went in.
  Nickname aliases that never appear adjacent ("Lizzy" for "Elizabeth") can still
  slip through, so `sample` and `validate` are there to be run before publishing.
- **`character_swap` requires gender agreement** between the removed and inserted
  character. Without it, the surrounding pronouns disagree and the item becomes
  solvable by grammar rather than by continuity reasoning.

## Build order

1. ~~Error-injection script + clean/injected corpus~~ — done, with the yield caveat above.
1b. **LLM injector** for balanced error-type volume — the measured next step.
2. Baseline evals (frontier LLM full context, open LLM full context, naive Jev), publish benchmark v0.
3. Extraction pass (LLM) + story-state schema.
4. Jev verification pass wired to story-state.
5. LLM escalation/explanation pass.
6. Re-run the full benchmark with the hybrid pipeline; publish the cost-vs-F1 chart.
7. Static leaderboard site + CLI for a user-supplied manuscript.

## Notes

- `books.tsv` is the input list; titles there are labels only. `fetch` writes the
  title it actually found to `corpus/books_fetched.tsv`, so a wrong Gutenberg id
  surfaces instead of silently mislabeling a novel. All 35 ids verified correct.
- `corpus/raw/` and the built `.jsonl` are gitignored; rebuild with `fetch` + `build`.
  The build is deterministic given `--seed`.
- Jev is a proprietary hosted API. The pipeline code here is open source; the model
  is not. Nothing in this repo has called it yet, and the claimed pricing and
  latency in the project brief are unverified.
