# continuity-bench

A benchmark for detecting continuity errors in novel-length fiction, and (later) a
Jev-powered editing pipeline that tries to match full-context LLM accuracy at a
fraction of the cost.

**Status: Phase 1 built, no model has been run against it yet.** The corpus, the
evaluation harness, the adversarial checks and the leaderboard all work end to end.
The paid baselines need an API key (see below).

```bash
./run_all_checks.sh      # self-checks, deterministic rebuild, validation, leakage attack, site
```

## What exists

Three stdlib-only files. The only optional dependency is the `anthropic` SDK, imported
lazily and needed solely for the paid predictor.

```bash
python3 bench.py fetch        # download the novels in books.tsv from Project Gutenberg
python3 bench.py build        # inject labeled errors -> corpus/continuity_v0.jsonl
python3 bench.py validate     # check every label, span and pair
python3 bench.py stats        # label/split/scope distribution
python3 bench.py sample --rule character_swap -n 5    # eyeball items with the edit marked
python3 bench.py export       # release files, with test labels held back

python3 evaluate.py list                      # registered predictors
python3 evaluate.py attack                    # every free heuristic vs the corpus
python3 evaluate.py run llm --limit 20 --yes   # a paid model run (needs credentials)

python3 build_site.py         # regenerate site/index.html from results/*.json
```

Current corpus: **272 matched pairs / 544 items from 27 novels.** Every injected item
ships with the same passage unedited as its control, so the false-positive rate is
measured on exactly the same text distribution as the recall. Splits are per novel, so
no novel appears in both `dev` and `test`. The build is deterministic given `--seed` —
the same inputs produce a byte-identical corpus.

## Why the headline metric is not F1

On a balanced paired corpus, **flagging every passage scores precision 0.500, recall
1.000 and therefore F1 0.667 while discriminating nothing at all.** Three of the free
heuristics land in exactly that band. F1 is reported, but the metric that means
something here is

> **J = recall − false-positive rate** (Youden's J) — 0 for any constant strategy, 1 for
> a perfect one.

Balanced accuracy is deliberately *not* reported: it is exactly `(1 + J) / 2`, so it
would be a second column that can never disagree with the first.

The project brief asks for a cost-vs-F1 chart as the release headline, and the site
draws it, with the 0.667 flag-everything line marked so no point below it can look good.

## Is the corpus actually measuring reasoning?

`evaluate.py attack` exists to answer that before anyone spends money. It runs every
zero-cost predictor, including two that model an adversary who downloaded the dataset:

| attack | J | what it exploits |
|---|---|---|
| `corpus_prior` | **+0.066** | pools every passage of a novel, then flags names that are locally rare but novel-wide common — exactly what `character_swap` creates |
| `singleton_name` | −0.011 | flags a passage holding a name mentioned once beside frequent names |
| `trait_disagreement` | +0.011 | flags two different colours describing one feature |
| `rarest_name` / `always_error` | +0.000 | constant strategies; F1 0.667, discrimination zero |
| `narration_person` | +0.000 | flags narration mixing first and third person (see the POV note below) |
| `pair_leak` *(diagnostic)* | +0.121, **localization 1.000** | diffs the two halves of a matched pair |
| `pair_diff_targeted` *(diagnostic)* | **+0.699**, FPR 0.000 | diffs the pair, then tests only the token that changed |

**The best genuine attack reaches J +0.066, so discrimination is not leaked** — an
injected intruder is statistically indistinguishable from the minor characters real
prose mentions once. Diagnostics are excluded from that verdict: they exploit the
release format rather than the text, and are kept registered so a mitigation cannot
quietly regress.

**The matched-pair design is itself an answer key, and that took three goes to close.**

1. `pair_leak` localizes with **loc 1.000** by diffing the two halves of a pair. First
   mitigation: hold back the test labels.
2. That wasn't enough — the internal `passage_id` ends in `-err` or `-clean`, so the
   export was publishing the answer for every test item *in the id itself*. Second
   mitigation: opaque hashed ids.
3. Still not enough. I claimed the residual was harmless because "without labels a
   submitter cannot tell which half carries the error, so detection stays at chance."
   **That claim was wrong, and measuring it proved it wrong:** diffing a pair hands over
   the exact token that changed, which turns `corpus_prior`'s blind search across a whole
   passage into one targeted test — the injected half is the one whose differing token
   appears once while its counterpart appears often. `pair_diff_targeted` scores
   **J +0.699 at FPR 0.000**, an order of magnitude past every blind attack.

Final mitigation: `bench.py export` ships **one half of each test pair and no more**, so
there is nothing left to diff (verified: zero recoverable pairs in the published file).
The cost is exact per-pair matching on the released test split — dev keeps both halves,
since its job is for humans to read — and 67 published test passages instead of 134. The
property that matters, a false-positive rate measured on the same passage pool as the
recall, survives in expectation because each pair contributes one half drawn from that
pool. `release/` is gitignored so the held-back labels cannot be published by accident.

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

`context_scope` is `passage` when the contradicted fact is stated in the same passage,
and `long_range` when it was established chapters earlier — the case that cannot be
solved by reading one passage, and the reason the Phase 2 story-state pipeline exists.

## The injection rules, and what they actually yield

| rule | error type | items | what it does |
|---|---|---|---|
| `character_swap` | fact_contradiction | 266 | puts a character who is absent from the scene into it, in place of one who is present |
| `trait_flip` | fact_contradiction | 3 | flips the later of two matching eye/hair colour phrases with the same owner |
| `timeline_weekday` | timeline | 3 | breaks a day sequence the passage itself states |

**The corpus is 98% one error type, and that is a finding rather than a bug to paper
over.** Measured across all 35 downloaded novels (~70,000 paragraphs):

| pattern | occurrences | per novel |
|---|---|---|
| day-of-week mentions | 924 | 26.4 |
| …forming a checkable `DAY … connective … DAY` sequence | 3 | 0.1 |
| eye/hair colour phrases | 425 | 12.1 |
| …repeated with the same owner, so a flip is a contradiction | 32 | 0.9 |
| month + season co-occurrence | 25 | 0.7 (and mostly the modal "may", not May) |
| `Name's <kinship>` | 1 | 0.0 |

Rule-based injection over public-domain fiction supports exactly one high-volume error
type plus a long tail. The facts regex can verify are not repeated often enough in real
prose. Balanced volume needs an LLM injector that *writes* the contradiction — that is
step 1b, and it is a measured conclusion rather than a guess.

## Why there is no POV-slip error type

The brief lists POV/tense slip as one of three candidate error types. It is
implementable and high-yield: switching a narrated `he/she <verb>` to `I <verb>` in a
third-person novel gives **17,258 injection sites across 14 novels**, and a build
produced 180 items — which would have taken the corpus from 98% one error type to a
59/40 split across two.

**It was dropped because a five-line regex detects 100% of them.** The
`narration_person` attack finds every injected item (recall 1.000) and pushed the
overall free-heuristic J from +0.066 to **+0.350**, past the leakage threshold. The
reason is structural: the injected "I" is the only first-person narration token in an
otherwise pure third-person passage, so it is a lexical outlier rather than a
contradiction anything has to reason about. Matching the clean controls on that
statistic is not available either — third-person novels sit at a 0.003 narrated-"I"
ratio, so passages that already contain one barely exist.

With the rule removed, `narration_person` collapses to J +0.000 (recall 0.456 against
FPR 0.456 — noise), confirming it was detecting only the artifact.

The conclusion is worth more than the 180 items: **a POV slip is surface-detectable
and belongs in a linter, not in a benchmark meant to measure state tracking.** The
error types that need a story-state model are the ones where the contradicted fact is
elsewhere in the text. That is an argument for the Phase 2 premise, and an argument
against one third of the brief's error-type list.

## Running the paid baselines

Needs credentials: `ANTHROPIC_API_KEY`, or an `ant auth login` profile the SDK picks up
on its own. `pip install anthropic`.

```bash
python3 evaluate.py run llm --split dev --limit 20 --yes                  # smoke test, ~40 calls
python3 evaluate.py run llm --model claude-opus-5 --effort high --yes     # full dev split
python3 evaluate.py run llm --model claude-haiku-4-5 --effort low --yes   # the cheap end
```

A paid run refuses to start without `--yes`, and prints the item count first. Cost comes
from `response.usage` against the published per-million-token rates, not an estimate.
Responses are constrained with a JSON schema; a response that still fails to parse is
counted and reported rather than silently scored as "no error".

The predictor's system prompt states that about half the passages are clean. That is
true of this corpus and stops a model's prior from dominating the result, but it is a
prompt choice worth recording — a deployment on a real manuscript has a far lower base
rate.

## Known limitations

- **Training-data contamination.** All 35 novels are famous and certainly in every
  frontier model's training data; a model may "remember" the original text rather than
  reason about the passage. `recall_by_novel` is reported in every results JSON so
  memorisation shows up as per-novel variance. Swapping in obscure Gutenberg texts is
  the real fix and is not done yet.
- **Injected errors are not natural errors.** The benchmark measures detection of
  *injected* errors and the README should keep saying so.
- **No coreference.** `trait_flip` trusts the owner token ("her blue eyes") as the whole
  referent check. That holds inside a passage and fails across a novel, so pronoun-owned
  traits are restricted to `passage` scope and only possessive names ("Anna's hair") are
  allowed at long range. Two women in one passage both described as "her blue eyes"
  would still be mislabeled.
- **Name detection is heuristic.** A name qualifies as a person if it appears capitalised
  mid-sentence, *speaks* at least twice, rarely follows an article, and stands alone a
  fair share of the time. Every one of those filters exists because of a real failure:
  `The`, `Master`, `She`, `Company`, `Mars`, `Mrs`, `Sir`, `Van` (from "Van Helsing") and
  `York` (from "New York") all got injected as characters before they went in. Nickname
  aliases that never appear adjacent ("Lizzy" for "Elizabeth") can still slip through,
  which is what `sample` and `validate` are for.
- **The stoplist is only for closed classes.** Measured over the 35 novels, just **12 of
  289** stoplist entries ever rejected anything (`She`, `They`, `You` and a handful of
  vocatives), and none of the geography, nationalities or nobility titles did — the
  distributional filters already handle those. Those 22 words are gone and a test now
  asserts the filters reject `London`, `Mars` and `Company` with no list help, so the
  answer to "should I add a word?" is "no, fix a filter." The residue is irreducible: a
  term of address sits in exactly the syntactic slot a name sits in, so no test on
  capitalisation can reach it.
- **`character_swap` enforces gender agreement** between the removed and inserted
  character, and refuses an intruder who is an alias of the victim or of anyone in the
  scene. Without the first, surrounding pronouns disagree and the item is solvable by
  grammar; without the second, `Sawyer` replaces `Tom` and the "error" is not one.
- **Only two error types have meaningful volume**, and POV slip was tried and rejected
  on evidence — see below.

## Build order

1. ~~Error-injection script + clean/injected corpus~~ — done, with the yield caveat above.
1b. **LLM injector** for balanced error-type volume — the measured next step.
2. ~~Evaluation harness, metrics, adversarial floor checks, leaderboard~~ — done.
   **Paid baselines still to run** (frontier LLM, open LLM, naive Jev).
3. Extraction pass (LLM) + story-state schema.
4. Jev verification pass wired to story-state.
5. LLM escalation/explanation pass.
6. Re-run the full benchmark with the hybrid pipeline; publish the cost-vs-F1 chart.
7. CLI for a user-supplied manuscript.

## Notes

- `books.tsv` is the input list; titles there are labels only. `fetch` writes the title it
  actually found to `corpus/books_fetched.tsv`, so a wrong Gutenberg id surfaces instead
  of silently mislabeling a novel. All 35 ids verified correct.
- `corpus/raw/`, the built `.jsonl` and `release/` are gitignored. Rebuild with `fetch`
  and `build`.
- The build takes ~23s for 35 novels. About 70% of that is the two candidate-scan
  regexes (`candidates_trait_flip`, `candidates_timeline`) dragging lazy `.{0,400}?`
  spans across each whole novel. Pre-filtering on a cheap anchor token (`eyes?|hair`,
  or the weekday alternation) and running the expensive patterns only near hits would
  bring it into single digits. Not done: the build is run rarely and correctness of the
  labels mattered more than its wall clock.
- Jev is a proprietary hosted API. The pipeline code here is open source; the model is
  not. Nothing in this repo has called it yet, and the pricing and latency in the project
  brief are unverified — worth confirming before Phase 2 design depends on them.
