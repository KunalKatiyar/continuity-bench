# continuity-bench

[Live leaderboard](https://kunalkatiyar.github.io/continuity-bench-leaderboard/) ·
[leaderboard repo](https://github.com/KunalKatiyar/continuity-bench-leaderboard)

Can a model tell when a novel contradicts itself? And what does one catch cost?

This is a benchmark that injects continuity errors into public-domain fiction at known
locations, plus a three-stage editing harness built around Jev, TypeSafe's non-generative
classifier. The bet behind the harness: checking "does this paragraph contradict a fact
we already know?" is a closed question, and closed questions are cheap. Only the ends
need a real LLM.

```bash
./run_all_checks.sh     # 112 self-checks, deterministic rebuild, validation, leakage attack, site
```

## Where it stands

The corpus, the harness, the adversarial checks and the leaderboard all work. One model
has been run against it end to end. The Jev runs are written and tested but unrun,
because TypeSafe closed new signups before I could get a key.

The first baseline is not flattering to the model. `llama3.1:8b`, full 410-item dev
split, greedy decoding:

| J | 95% CI | recall | FPR | F1 | localization |
|---|---|---|---|---|---|
| +0.005 | −0.039 to +0.048 | 0.981 | 0.976 | 0.663 | 0.114 |

It flags 98% of the injected passages and 98% of the clean ones. That is the
"say yes to everything" strategy with extra steps, and its F1 of 0.663 lands on the
flag-everything score of 0.667 exactly as you would expect. Localization is 0.114
against a chance rate near 0.083, so it isn't finding the right paragraph either.

That result is the clearest argument for the metric choice below.

## F1 is the wrong headline here

Every injected passage in this corpus is paired with the same passage left alone. On a
balanced paired set, flagging everything gets you precision 0.500, recall 1.000, and so
F1 0.667, while separating nothing whatsoever. An 8B model scored precisely that. A
leaderboard reporting F1 alone would have shown it as roughly competitive.

So the number that leads is **J = recall − false-positive rate**. Zero for any constant
strategy, 1 for a perfect one. F1 is still in the table, with the 0.667 line drawn on the
chart so nothing below it can look respectable.

Balanced accuracy is deliberately absent. It works out to exactly `(1 + J) / 2`, so it
would be a column that can never disagree with the one beside it.

Every run also carries a 95% Wilson interval, and the leaderboard distinguishes two
things that look identical if you only report significance: an interval that is tight
around zero (the model genuinely doesn't discriminate) and one that is wide around zero
(you didn't run enough items to know). My first 20-pair run was the second kind, ±0.23,
and I nearly reported it as a finding.

## Does the benchmark measure reasoning, or just leak?

`evaluate.py attack` answers that before anyone spends money, by throwing every
zero-cost heuristic at the corpus, including two that model someone who downloaded the
whole dataset.

| attack | J | what it exploits |
|---|---|---|
| `corpus_prior` | **+0.066** | pools novel-wide name counts, flags names that are locally rare but common elsewhere in the book |
| `trait_disagreement` | +0.011 | two different colours describing one feature |
| `narration_person` | +0.000 | narration mixing first and third person |
| `rarest_name`, `always_error` | +0.000 | constant strategies; F1 0.667, discrimination zero |
| `singleton_name` | −0.011 | a name mentioned once beside frequent ones |
| `pair_leak` *(diagnostic)* | +0.121, loc **1.000** | diffs the two halves of a matched pair |
| `pair_diff_targeted` *(diagnostic)* | **+0.699**, FPR 0.000 | diffs the pair, then tests only the changed token |

The best honest attack gets J +0.066. An injected intruder turns out to be
statistically indistinguishable from the minor characters real prose mentions once,
which is the property that makes the corpus worth running a model against.

### The matched-pair design is an answer key, and I got it wrong twice

Diagnostics are listed separately because they cheat the release format rather than read
the text. Closing that took three attempts:

First I held back the test labels. Not enough: the internal `passage_id` ends in `-err`
or `-clean`, so the export was publishing the answer in the id itself.

Then I made the ids opaque hashes, and argued the rest was harmless, because without
labels you still can't tell which half of a pair carries the error. Detection stays at
chance, I said.

That was wrong, and measuring it is what showed me. Diffing a pair hands you the exact
token that changed, which turns a blind search across a whole passage into one targeted
test: the injected half is the one whose changed token appears once while its counterpart
appears often. `pair_diff_targeted` scores **J +0.699 at FPR 0.000**, an order of
magnitude past anything working blind.

The fix is structural. `bench.py export` ships one half of each test pair and no more, so
there is nothing left to diff, verified by checking that zero pairs survive in the
published file. Both attacks stay registered so the mitigation can't quietly regress.
Cost: 67 published test passages instead of 134, and no exact per-pair matching on the
test split. Dev keeps both halves, since its job is for humans to read.

## The Jev editing harness

Phase 2, in `pipeline.py`:

```
chapter text  --[LLM, once per chapter]-->  story state
paragraph     --[Jev, once per paragraph]-> contradicts state? yes/no + confidence
flagged only  --[LLM, once per flag]------> confirm, write the author's note
```

Verification is closed: does this paragraph contradict something already in state? That
is what a System One model answers in one parallel pass, for input tokens alone, at
$0.042 per million with no output charge. Generation is only needed to build the state
and to explain a real hit. So the expensive model touches a small slice of the text.

Escalation keys on Jev's calibrated confidence, not just its answer. A confident "no" is
dropped. An unsure "no" gets escalated, because unsure is exactly where a cheap gate
fails, and silently dropping those would hide the pipeline's real failure mode.
`escalation_rate` is recorded per run, and it is the number the whole cost argument rests
on.

Two entries reach the leaderboard: `jev`, the naive Jev-only pass the brief asks to be
reported honestly, and `jev_hybrid`, the actual proposal.

```bash
.venv/bin/python evaluate.py run jev        --split dev --yes   # ~$0.03 for 410 items
.venv/bin/python evaluate.py run jev_hybrid --split dev --yes
```

Credentials come from `TYPESAFE_API_KEY` or `~/.typesafe-key`.

### Running the architecture without Jev

Signups are closed, so neither of those can run here yet. `pipeline.LocalVerifier` is a
drop-in with the same interface, and `hybrid_local` runs all three stages on local models
with no keys:

```bash
.venv/bin/python evaluate.py run hybrid_local --model llama3.1:8b --split dev --limit 10
```

This tells you the harness works on real items, what the escalation rate is, and whether
gate-plus-escalation beats one whole-passage pass by the same model. Every interface Jev
touches gets exercised, so swapping it back in is one line.

It tells you nothing about Jev's accuracy or calibration, and nothing about cost. A local
gate costs a model call per paragraph, which is the exact cost Jev exists to remove, and
a generative model's self-reported confidence is not a calibrated probability. Runs
record `projected_jev_gate_usd`, which applies TypeSafe's published rate to measured gate
tokens. A projection, labelled as one.

## The NLI framing, and what it found

A contradiction between two paragraphs is natural language inference: premise = what
the story established, hypothesis = the new paragraph. `nli.py` tests that with an
off-the-shelf MNLI encoder, no prompting and no API, scoring every ordered pair of
paragraphs in a passage and taking the maximum.

Full dev split, 27,018 pairs, `DeBERTa-v3-base-mnli-fever-anli`:

| rule | AUC | n | reading |
|---|---|---|---|
| `character_swap` | **0.5018** | 402 | chance. no separation at all |
| `trait_flip` | 0.5556 | 6 | no statistical power |
| `timeline_weekday` | 1.0000 | 2 | no statistical power |

AUC is reported because it needs no threshold: it says whether the classes can be
separated *at any* cut point, which J and F1 cannot.

**`character_swap` is not an NLI contradiction, and that is the finding.** "Elizabeth
walked in" and "Wickham crossed the room" are not contradictory propositions; both can
be true. The error is that Wickham is not *in the scene* — a presupposition failure
that needs state tracking, not entailment.

I suspected truncation was confounding this, since 22% of pairs hit the old 256-token
limit and `longest_first` can clip the hypothesis. Fixed it (512 tokens, hypothesis
clipped explicitly so the premise is what gives) and re-ran: AUC moved 0.5031 → 0.5024.
Not a truncation artifact.

The uncomfortable conclusion is that **the corpus cannot currently tell "models are bad
at continuity" apart from "we asked the wrong question about the wrong error type".**
98% of it is an error type that does not match the question every predictor is asked,
and the 2% that does match has n=6. That reframes every J ≈ 0 on the board, and it is
the strongest argument yet for the LLM injector: regex can only mass-produce the one
error type that fits the question worst.

## An open-weights System One gate

TypeSafe closed Jev signups, but [Laya](https://huggingface.co/convaiinnovations/laya)
is the same category of model — typed questions over a state block, never generates,
calibrated probabilities — under Apache 2.0, 421M on a ModernBERT-large backbone,
running locally. It drops into the `JevVerifier` slot unchanged, and unlike Jev it can
be fine-tuned.

Its 512-token budget (~320 for state) is smaller than a benchmark passage, which turns
out to argue *for* the architecture rather than against it: you cannot hand it a
chapter, you must hand it a compact extracted state.

| run | J | 95% CI | recall | FPR | wall clock |
|---|---|---|---|---|---|
| `laya_english` | +0.010 | ±0.087 | 0.117 | 0.107 | **2m15s** |
| `laya_typed-decisions` | +0.000 | ±0.045 | 0.024 | 0.024 | ~2m |

Both measured nulls, which is the predicted result for a naive pass over a raw passage.
The number worth keeping is the wall clock: 410 items in 2m15s against `llama3.1:8b`'s
44 minutes, ~20x. That is the first vendor latency claim in this project that survived
an independent check.

Two things found by running it that the model card does not say. The shipped checkpoint
warns at load that it carries an invalid temperature for 11-option choice questions —
the yes/no path used here is unaffected, but calibration is Laya's headline claim and
the warning is in the artifact. And the `typed-decisions` checkpoint, the fine-tuned one
carrying their 0.766 accuracy claim, did *worse* here than the base model, flagging 2.4%
of anything. Their headline number does not transfer to this task.

## Two declared gate questions

The NLI result said the corpus's dominant error type is presupposition failure, not
contradiction. Asking the gate about presence as well as contradiction is the free
experiment that tests it — so it is a declared variant, not an improved wording:

```bash
.venv/bin/python evaluate.py run laya_hybrid --question contradiction --split dev
.venv/bin/python evaluate.py run laya_hybrid --question presupposition --split dev
```

Swapping the wording in place would have invalidated every recorded run and confounded
a gate comparison with a question change — which is the rule this project already
applies to providers ("two providers judged on two different prompts is not a
comparison"). The delta between the two variants on the same items is the measurement
that separates "models are bad at continuity" from "we asked the wrong question".

## The human baseline

Every approach on the board scores J near zero, and two very different explanations fit:
the models are bad, or the items are not solvable by a careful reader. Nothing else
measures the difference.

```bash
python3 human_baseline.py --n 40      # ~20 minutes, resumable
python3 human_baseline.py --score     # writes a leaderboard row
python3 human_baseline.py --review    # what was missed, with the injection explained
```

It shows one half of each matched pair and never both — the halves differ by a single
token, so a reader shown both is running the `pair_diff_targeted` attack by hand. No
feedback during a run, or it measures how fast someone learns the generator. Time per
passage is recorded, because a reader who needs four minutes is a different proposition
from one who needs twenty seconds, and the pipeline exists to replace the expensive one.

The row publishes as `kind: human`, not `model`, and is kept off the cost chart. A human
is the most expensive predictor here; rendering them at $0.00 would make them its
cheapest point, which is the same failure this project refuses for an unlisted model's
rate. Their cost axis is seconds per passage.

## The corpus

272 matched pairs, 544 items, 27 novels. Deterministic given `--seed`: same inputs,
byte-identical output, which held across Python 3.12 and 3.13.

```bash
python3 bench.py fetch      # pull the novels in books.tsv from Project Gutenberg
python3 bench.py build      # inject labeled errors -> corpus/continuity_v0.jsonl
python3 bench.py validate   # check every label, span and pair
python3 bench.py sample --rule character_swap -n 5   # eyeball items with the edit marked
python3 bench.py export     # release files, test labels held back
```

| rule | error type | items | what it does |
|---|---|---|---|
| `character_swap` | fact_contradiction | 266 | drops a character who is absent from the scene into it, replacing one who is present |
| `trait_flip` | fact_contradiction | 3 | flips the later of two matching eye/hair colour phrases with the same owner |
| `timeline_weekday` | timeline | 3 | breaks a day sequence the passage states itself |

Yes, that is 98% one error type, and no, it isn't for want of trying. Across all 35
downloaded novels, roughly 70,000 paragraphs:

| pattern | occurrences | per novel |
|---|---|---|
| day-of-week mentions | 924 | 26.4 |
| …forming a checkable `DAY … connective … DAY` sequence | 3 | 0.1 |
| eye/hair colour phrases | 425 | 12.1 |
| …repeated with the same owner, so a flip contradicts something | 32 | 0.9 |
| month + season co-occurrence | 25 | 0.7 (mostly the modal "may", not May) |
| `Name's <kinship>` | 1 | 0.0 |

Regex injection over real prose supports one high-volume error type and a long tail. The
facts a regex can verify simply aren't repeated often enough in actual novels. Balanced
volume needs an LLM that writes the contradiction instead of pattern-matching one, which
is the next step.

### Why there's no POV-slip error type

The brief lists POV/tense slip as a candidate, and it works: switching a narrated
"he/she <verb>" to "I <verb>" in a third-person novel gives 17,258 injection sites across
14 novels, and a build produced 180 items. That would have taken the corpus from 98% one
type to a 59/40 split.

I dropped it because five lines of regex detect 100% of them. The `narration_person`
attack catches every injected item and pushed the overall free-heuristic J from +0.066 to
+0.350. The reason is structural: the injected "I" is the only first-person narration
token in an otherwise pure third-person passage, so it's a lexical outlier, not a
contradiction anyone has to reason about. Matching the controls on that statistic isn't
available either, since third-person novels sit at a 0.003 narrated-"I" ratio.

Remove the rule and `narration_person` collapses to J +0.000, which confirms it was only
ever detecting the artifact.

The conclusion is worth more than the 180 items: a POV slip is surface-detectable and
belongs in a linter, not in a benchmark meant to measure state tracking.

## Running models

Four providers, one shared prompt and one shared JSON schema. Two providers judged on
two different prompts is not a comparison, so they all subclass `VerdictPredictor` and
implement only the API call.

```bash
~/.pyenv/versions/3.12.10/bin/python -m venv .venv      # a bare python3 here resolves
.venv/bin/python -m pip install openai anthropic        # to a sibling project's venv

.venv/bin/python evaluate.py run local  --model llama3.1:8b --split dev      # free, ~44 min
.venv/bin/python evaluate.py run openai --model gpt-4o-mini --limit 20 --yes # ~$0.01
.venv/bin/python evaluate.py run llm    --model claude-opus-5 --split dev --yes
```

Keys resolve from `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, then `~/.openai-key` /
`~/.anthropic-key`, so nothing has to be pasted into a terminal.

Cost of a full dev pass (410 items, ~607k input tokens): Jev $0.03, `gpt-4o-mini` $0.11,
`gpt-4.1` $1.48, `claude-haiku-4-5` $1.63, `claude-sonnet-5` $3.26, `claude-opus-5` $8.16.
Local models are free but not fast: `llama3.1:8b` runs 6.4 s/item on an RTX 1000.

Spending guards, since this costs money: a paid run refuses to start without `--yes` and
prints the item count first. Cost comes from the API's own usage, never an estimate. An
unlisted model reports $0.00 loudly rather than inventing a rate, because a wrong rate
quietly corrupts the cost axis this whole project is about. Fatal errors like quota or
auth stop the run immediately with one line instead of 400 tracebacks; transient ones
skip a single item, still count against recall, and get reported.

One more thing learned the hard way: local runs pin temperature and seed. Before that,
the identical 40 items scored J −0.100 and then J +0.250 on consecutive runs. Hosted
frontier models reject sampling parameters, which is the other reason every run ships
with a confidence interval.

## What this doesn't show

Injected errors are not natural errors. This measures detection of programmatically
injected contradictions, and nothing else.

The novels are famous, so they're in every frontier model's training data. A model might
remember that the eyes were blue rather than reason about the passage. Per-novel recall
is in every result file so memorisation shows up as variance. Swapping in obscure texts
is the real fix, and it isn't done.

`trait_flip` trusts the owner token ("her blue eyes") as its entire referent check. That
holds inside a passage and fails across a novel, so pronoun-owned traits are restricted
to passage scope and only possessive names ("Anna's hair") are allowed at long range. Two
women in one passage both described as "her blue eyes" would still be mislabeled.

Name detection is heuristic. A name counts as a person if it appears capitalised
mid-sentence, speaks at least twice, rarely follows an article, and stands alone a fair
share of the time. Each of those filters exists because of a real failure: `The`,
`Master`, `She`, `Company`, `Mars`, `Mrs`, `Sir`, `Van` (from "Van Helsing") and `York`
(from "New York") all got injected as characters before they went in. Nicknames that
never appear adjacent, "Lizzy" for "Elizabeth", can still slip through.

The stoplist is only for closed classes. Of 289 entries, 12 ever rejected anything, and
none of the geography, nationalities or nobility titles did; the distributional filters
already handle those. Those 22 words are gone and a test now asserts that `London`, `Mars`
and `Company` are rejected with no list help, so the answer to "should I add a word?" is
"no, fix a filter."

The build takes about 23 seconds for 35 novels, roughly 70% of it two candidate-scan
regexes dragging lazy spans across whole novels. Pre-filtering on a cheap anchor token
would bring it into single digits. Not done: correctness of the labels mattered more than
the wall clock.

## Notes

`books.tsv` is the input list, and its titles are labels only. `fetch` writes the title it
actually found to `corpus/books_fetched.tsv`, so a wrong Gutenberg id surfaces instead of
silently mislabeling a novel. All 35 ids check out.

`corpus/raw/`, the built `.jsonl` and `release/` are gitignored. Rebuild with `fetch` and
`build`.

Jev is proprietary and hosted (TypeSafe AI, released 2026-09-15). The pipeline code here
is open; the model isn't.
