# Can a model tell when a novel contradicts itself?

I've been building a benchmark for continuity errors in fiction — the kind where a
character's eyes change colour between chapters, or someone walks into a room they
were never in. Then an editing pipeline that tries to catch them cheaply.

Nothing works yet. That's most of what I have to report, and it's the interesting part.

## The bet

Continuity checking looks like it should be expensive. A 100,000-word novel is about
1,300 paragraphs, and every one of them could contradict anything that came before. If
you need a frontier model to read each paragraph against everything established so
far, you're paying frontier prices 1,300 times.

But the *question* is cheap. "Does this paragraph contradict a fact we already know?"
is a yes/no with a closed answer set. That's not what generative models are for — it's
what classifiers are for.

So: an LLM reads each chapter once and extracts a story state. A cheap classifier
checks every paragraph against that state. The LLM only comes back for the handful the
classifier flags, to confirm and write the note a human reads.

That's the architecture. The whole thing rests on the classifier being both cheap and
good enough, which is an empirical question, which is why I built the benchmark first.

## Building something to measure it

The corpus injects known errors into public-domain novels at known locations. Every
injected passage ships alongside **the same passage unedited**, so the false-positive
rate is measured on exactly the same text as the recall. Splits are per novel, so no
book appears in both dev and test.

Then I spent a while attacking it, because a benchmark whose errors a regex can find
isn't measuring reasoning. The strongest zero-cost heuristic gets J +0.066 — near
enough to nothing. An injected intruder turns out to be statistically indistinguishable
from the minor characters real prose mentions once, which is the property that makes
the thing worth running a model against.

## Seven approaches, all at zero

| | J | 95% CI |
|---|---|---|
| llama3.1:8b | +0.005 | ±0.044 |
| qwen2.5:7b | +0.000 | ±0.070 |
| qwen2.5vl:7b | −0.005 | ±0.023 |
| DeBERTa-v3 MNLI (zero-shot) | −0.005 | ±0.083 |
| Laya (System One) | +0.010 | ±0.087 |
| Laya, fine-tuned checkpoint | +0.000 | ±0.045 |
| Laya hybrid pipeline | +0.000 | ±0.088 |

J is recall minus false-positive rate. Zero means the run separates nothing.

## Four things worth knowing

**F1 would have hidden all of this.** On a corpus where every injected passage is paired
with a clean twin, flagging *everything* scores precision 0.500, recall 1.000, and
therefore F1 0.667 — while discriminating nothing whatsoever. llama3.1:8b scored 0.663.
It flags 98% of injected passages and 98% of clean ones. A leaderboard reporting F1
alone would have shown it as roughly competent. I nearly published that.

**The corpus may be asking the wrong question.** I tested the task as natural language
inference — premise is what's established, hypothesis is the new paragraph, label is
contradiction. Across 27,000 paragraph pairs, the dominant error type scored AUC 0.5018
at n=402. Chance, to three decimals.

The reason is structural, and I think it's the most useful thing I've found. "Elizabeth
walked in" and "Wickham crossed the room" are not contradictory propositions. Both can
be true. The error is that Wickham isn't *in the scene* — presupposition failure, not
contradiction. It needs state tracking, and I'd been asking every model about
contradiction. Which means the benchmark currently cannot distinguish "models are bad
at this" from "I asked the wrong question about the wrong error type."

**I built an error type and then deleted it.** POV slips — switching a narrated "he
walked" to "I walked" in a third-person novel — gave 17,258 injection sites and would
have taken the corpus from 98% one error type to a 59/40 split. Five lines of regex
detect 100% of them. The injected "I" is a lexical outlier, not something anyone has to
reason about. A POV slip belongs in a linter, not a benchmark about state tracking.

**The matched-pair design was an answer key, and I got it wrong twice.** Holding back
the test labels wasn't enough — the internal passage IDs ended in `-err` and `-clean`,
publishing the answer in the ID. Opaque hashes weren't enough either: I argued the rest
was harmless because without labels you can't tell which half carries the error.
Measuring it proved me wrong. Diffing a pair hands you the exact changed token, which
turns a blind search into one targeted test. That attack scores **J +0.699** against
+0.066 for anything working blind. The release now ships one half of each test pair.

## The classifier is fast, at least

Laya is an open-weights System One model — typed questions over a state block, never
generates, returns calibrated probabilities. It ran 410 passages in **2 minutes 15
seconds**, against 44 minutes for an 8B chat model. Roughly 20×, and the first vendor
latency claim in this project that survived an independent check.

The full pipeline runs too: the gate checked 960 paragraphs, escalated 197 of them
(20.5%), and the expensive model never touched the other 79.5%. The cost mechanism
works exactly as designed. The escalator then confirmed none of the 197. So I have a
pipeline that saves money reliably while finding nothing, which is not yet a product.

Two things the model card doesn't mention, found by running it: the shipped checkpoint
warns that it carries an invalid temperature for 11-option questions (the yes/no path I
use is clean), and their *fine-tuned* checkpoint — the one carrying the headline
accuracy number — does worse on this task than the base model.

## What I don't know yet

Whether a careful human can solve these items at all.

That's the fork everything hangs on. If a human scores well, the models are genuinely
bad and the benchmark is sound. If a human scores near chance, the items are unfair and
every number above is measuring nothing. I've built the tool to find out and haven't
run it.

I'd rather publish that gap than paper over it, because the gap is the whole reason this
exists. Nobody selling AI manuscript tools publishes accuracy numbers. Not one. The
pitch is "catches continuity issues in seconds" and there is no recall figure, no
false-positive rate, no methodology. Having spent a few weeks trying to measure this
honestly, I think I know why: it's harder than it looks, and the honest numbers are
unflattering.

Which is the actual argument for building the yardstick before the product.

---

Code, corpus builder and evaluation harness:
[continuity-bench](https://github.com/KunalKatiyar/continuity-bench).
Live leaderboard: [kunalkatiyar.github.io/continuity-bench-leaderboard](https://kunalkatiyar.github.io/continuity-bench-leaderboard/).
Every number is reproducible from `run_all_checks.sh` plus the command recorded in each
result file.
