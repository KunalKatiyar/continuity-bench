#!/usr/bin/env python3
"""Corpus builder for the novel continuity benchmark.

Subcommands
    fetch   download the public-domain novels listed in books.tsv from Project Gutenberg
    build   find injectable spots, inject labeled continuity errors, emit paired JSONL
    stats   summarise a built corpus
    sample  print items with the injected span marked, for manual spot-checking

Every injected item ships with a matched clean control: the same passage with no
edit, so false-positive rate is measured on exactly the same text distribution.

ponytail: injection is three narrow regex rules and no LLM anywhere. That is the
whole ceiling of this file - realism and yield are bounded by the rules below.
Add rules, or an LLM rewriter behind the same Hit interface, when the corpus runs
thin or the errors read as too mechanical.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "corpus" / "raw"
BOOKS_TSV = ROOT / "books.tsv"
FETCHED_TSV = ROOT / "corpus" / "books_fetched.tsv"
DEFAULT_CORPUS = ROOT / "corpus" / "continuity_v0.jsonl"
USER_AGENT = "continuity-bench/0.1 (research corpus builder)"

START_MARKERS = (
    "*** START OF THE PROJECT GUTENBERG EBOOK",
    "*** START OF THIS PROJECT GUTENBERG EBOOK",
)
END_MARKERS = (
    "*** END OF THE PROJECT GUTENBERG EBOOK",
    "*** END OF THIS PROJECT GUTENBERG EBOOK",
)

DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

CONNECTIVES = (
    (r"(?:the\s+)?(?:next|following)\s+(?:day|morning|evening|afternoon|night)", 1),
    (r"(?:the\s+)?(?:day|morning|evening|night)\s+(?:before|previous)", -1),
    (r"(?:the\s+)?(?:next|following)\s+week", 7),
    (r"a\s+week\s+(?:later|afterwards|after)", 7),
    (r"two\s+days\s+(?:later|afterwards|after)", 2),
    (r"three\s+days\s+(?:later|afterwards|after)", 3),
)

TRAIT_VALUES = {
    "eye": ("blue", "brown", "black", "grey", "gray", "green", "hazel"),
    "hair": ("fair", "dark", "black", "brown", "golden", "red", "grey", "gray", "white", "auburn", "flaxen"),
}

OWNER = r"(?:[Hh]is|[Hh]er|[Tt]heir|[Mm]y|[Yy]our|[A-Z][a-z]+['\u2019]s)"
NAME_OWNER = re.compile(r"^[A-Z][a-z]+['\u2019]s$")

HONORIFICS = {"Mr", "Mrs", "Ms", "Dr", "St", "Capt", "Col", "Gen", "Rev", "Prof", "Lt", "Sgt"}

# Only closed classes belong in NOT_NAMES. Measured over the 35 novels, 12 of these
# entries ever reject anything - "She", "They", "You" and a handful of vocatives - and
# they are irreducible: a term of address sits in exactly the syntactic slot a name
# sits in, so no distributional test on capitalisation can reach it. Places,
# nationalities and nobility were in this list and were all inert: person_names'
# speech, article and standalone filters already reject "London", "York", "Mars",
# "Company" and "Sir" with no help. Reach for a filter, not a new word, when something
# slips through.
CLOSED_CLASS = """
    about above after again against all almost alone along already also although always among and
    another any anyone anything are around because been before being below beside besides between
    both but came can cannot come could did does doing done down during each either else enough even
    ever every everyone everything except far few first for from further gave get give going gone
    good great had has have having her here hers herself him himself his how however indeed inside
    instead into its itself just last least less let like little long made make many may might mine
    more most much must myself near neither never next nobody none nor not nothing now off often
    once one only onto other others ought our ours ourselves out outside over own perhaps quite
    rather really said same seen shall she should since some somebody someone something sometimes
    soon still such sure take taken than that the their theirs them themselves then there these they
    thing think this those though thought through thus till together too took toward under unless
    until upon very was way well went were what whatever when whenever where whether which while who
    whom whose why will with within without would yes yet you your yours yourself
"""

TERMS_OF_ADDRESS = """
    Almighty Aunt Captain Colonel Doctor Father God Heaven Lady Lord Madam Madame Mama
    Mamma Master Mistress Miss Monseigneur Monsieur Mother Papa Parson Providence
    Reverend Signor Sir Sire Squire Uncle
"""

CALENDAR = """
    January February March April May June July August September October November
    December Monday Tuesday Wednesday Thursday Friday Saturday Sunday
"""

NOT_NAMES = frozenset(
    [word.capitalize() for word in CLOSED_CLASS.split()]
    + TERMS_OF_ADDRESS.split()
    + CALENDAR.split()
    + sorted(HONORIFICS)
)

SPEECH_VERBS = (
    "said|cried|replied|answered|asked|exclaimed|murmured|whispered|continued|observed|returned"
)
PERSON_AS_OBJECT = re.compile(r"\b(?:" + SPEECH_VERBS + r")\s+(?P<name>[A-Z][a-z]{2,})\b")
PERSON_AS_SUBJECT = re.compile(r"\b(?P<name>[A-Z][a-z]{2,})\s+(?:" + SPEECH_VERBS + r")\b")
ADJACENT_NAMES = re.compile(r"\b(?P<first>[A-Z][a-z]{2,})\s+(?P<second>[A-Z][a-z]{2,})\b")
BLOCKED_BEFORE = re.compile(
    r"(?:\b(?:" + "|".join(sorted(HONORIFICS)) + r")\.?\s+|\b[A-Z][a-z]+\s+)$"
)
BLOCKED_AFTER = re.compile(r"^\.?\s+[A-Z][a-z]+")
DETERMINED = re.compile(r"\b(?:the|a|an|this|that|these|those|our|your)\s+$", re.I)
MALE_PRONOUNS = re.compile(r"\b(?:he|him|his|himself)\b", re.I)
FEMALE_PRONOUNS = re.compile(r"\b(?:she|her|hers|herself)\b", re.I)
PRONOUN_WINDOW = 80
MIN_SPEECH_EVIDENCE = 2
MAX_ARTICLED_SHARE = 0.1
MIN_STANDALONE_SHARE = 0.3
MIN_LOCAL_MENTIONS = 3
MIN_NOVEL_MENTIONS = 20
MAX_FIRST_PERSON_SHARE = 0.15

QUOTED_SPAN = re.compile("[\u201c][^\u201d]*[\u201d]|\"[^\"]*\"", re.S)
PERSON_INVARIANT_VERBS = (
    "had|was|could|would|should|might|must|went|came|saw|knew|felt|thought|took|stood"
    "|sat|began|heard|looked|turned|walked|waited|watched|remembered|understood"
)
THIRD_PERSON_NARRATION = re.compile(
    r"\b(?P<pron>he|she)\s+(?:" + PERSON_INVARIANT_VERBS + r")\b", re.I
)
FIRST_PERSON_NARRATION = re.compile(r"\bI\s+(?:" + PERSON_INVARIANT_VERBS + r")\b")

NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
QUOTE_ENDINGS = '.!?"”’\''


@dataclass
class Hit:
    """One injectable edit: replace text[start:end] with replacement.

    anchor is the offset of the earlier text the edit contradicts, and is what
    decides how wide a passage has to be for the error to be detectable at all.
    """

    rule: str
    error_type: str
    start: int
    end: int
    original: str
    replacement: str
    description: str
    anchor: int = 0
    long_range_ok: bool = False


def read_books():
    """Read books.tsv into a list of dicts with keys id, title, author."""
    with BOOKS_TSV.open(encoding="utf-8") as fh:
        return [row for row in csv.DictReader(fh, delimiter="\t") if row.get("id")]


def gutenberg_title(text):
    """Pull the Title: line out of a Gutenberg header, for metadata self-correction."""
    m = re.search(r"^Title:\s*(.+)$", text[:4000], re.M)
    return m.group(1).strip() if m else ""


def strip_boilerplate(text):
    """Drop the Gutenberg licence header and footer."""
    for marker in START_MARKERS:
        i = text.find(marker)
        if i != -1:
            nl = text.find("\n", i)
            text = text[nl + 1 :] if nl != -1 else text[i + len(marker) :]
            break
    for marker in END_MARKERS:
        i = text.find(marker)
        if i != -1:
            text = text[:i]
            break
    return text


def paragraphs(text, min_chars=80):
    """Split into whitespace-normalised prose paragraphs, dropping headings and short fragments."""
    out = []
    for block in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", block).strip()
        if len(para) < min_chars or para.isupper():
            continue
        if re.match(r"^(chapter|book|part|volume|act)\b", para, re.I) and len(para) < 200:
            continue
        out.append(para)
    return out


def join_paragraphs(paras):
    """Return (full_text, paragraph_start_offsets) so char offsets map back to paragraphs."""
    full = "\n\n".join(paras)
    starts, offset = [], 0
    for para in paras:
        starts.append(offset)
        offset += len(para) + 2
    return full, starts


def paragraph_of(starts, offset):
    """Index of the paragraph containing the given char offset."""
    return bisect.bisect_right(starts, offset) - 1


def name_counts_with_matches(text):
    """Count occurrences of likely proper names.

    A token qualifies as a name if it appears capitalised mid-sentence at least
    once; every occurrence of a qualifying token is then counted, including
    sentence-initial ones. Returns (counts, matches) - the match list is handed back
    so person_names can reuse the scan instead of making its own.

    ponytail: no NER model and no entity resolution. A capitalised adverb that
    happens to appear mid-sentence will be counted as a name; the count
    thresholds in rule_character_swap absorb that. Swap in spaCy or a
    coreference model if name precision starts costing corpus precision.
    """
    matches = list(NAME_RE.finditer(text))
    confirmed = set()
    for m in matches:
        token = m.group(0)
        if token in NOT_NAMES or token in confirmed:
            continue
        prev = text[max(0, m.start() - 8) : m.start()].rstrip()
        if not prev:
            continue
        if prev[-1] in QUOTE_ENDINGS:
            tail = prev[:-1].rsplit(None, 1)
            if not (prev[-1] == "." and tail and tail[-1] in HONORIFICS):
                continue
        confirmed.add(token)
    counts = Counter(m.group(0) for m in matches if m.group(0) in confirmed)
    return counts, matches


def name_counts(text):
    """Counts of likely proper names. See name_counts_with_matches."""
    return name_counts_with_matches(text)[0]


def _timeline_patterns():
    day_alt = "|".join(DAYS)
    for connective, offset in CONNECTIVES:
        yield (
            re.compile(
                r"\b(?P<first>" + day_alt + r")\b.{0,400}?\b" + connective + r"\b.{0,250}?\b(?P<second>" + day_alt + r")\b",
                re.S | re.I,
            ),
            offset,
        )


TIMELINE_PATTERNS = tuple(_timeline_patterns())


def candidates_timeline(text, rng):
    """Break a day sequence the passage itself states: 'Tuesday ... next morning ... Wednesday'."""
    for pattern, offset in TIMELINE_PATTERNS:
        for m in pattern.finditer(text):
            first = DAYS.index(m.group("first").capitalize())
            second = DAYS.index(m.group("second").capitalize())
            if (first + offset) % 7 != second:
                continue
            replacement = DAYS[(first + offset + 2) % 7]
            yield Hit(
                "timeline_weekday",
                "timeline",
                m.start("second"),
                m.end("second"),
                m.group("second"),
                replacement,
                f"the passage's own sequence puts this day at {m.group('second')}, not {replacement}",
                anchor=m.start("first"),
            )


def _trait_patterns():
    for noun, values in TRAIT_VALUES.items():
        value_alt = "|".join(values)
        noun_alt = r"eyes?" if noun == "eye" else r"hair"
        compound = "eyed" if noun == "eye" else "haired"
        yield noun, re.compile(
            r"(?P<owner>" + OWNER + r")\s+(?P<value>" + value_alt + r")\s+(?P<noun>" + noun_alt + r")\b"
        )
        yield noun, re.compile(
            r"(?P<owner>" + OWNER + r")\s+(?P<noun>" + noun_alt + r")\s+(?:were|was)\s+(?:a\s+)?(?P<value>" + value_alt + r")\b"
        )
        yield noun, re.compile(
            r"(?P<owner>" + OWNER + r")\s+(?P<value>" + value_alt + r")-(?P<noun>" + compound + r")\b"
        )


TRAIT_PATTERNS = tuple(_trait_patterns())


def same_colour(a, b):
    norm = {"gray": "grey"}
    a, b = a.lower(), b.lower()
    return norm.get(a, a) == norm.get(b, b)


def candidates_trait_flip(text, rng):
    """Flip the later of two matching trait phrases with the same owner token.

    The owner token is the only referent check there is, so it is trusted over a
    passage and distrusted over a novel: a pronoun owner is only accepted when both
    mentions sit inside one passage, while a possessive name ("Anna's hair") is
    accepted at any distance. Without that split, a long-range item pairs one
    woman's 'her black hair' with a different woman's, and is not an error at all.

    ponytail: that is still not coreference. Two women in the same passage both
    described as 'her blue eyes' remain a mislabeled item. Spot-check with
    `bench.py sample --rule trait_flip` before publishing, and add a real
    coreference pass if the residual rate matters.
    """
    groups = defaultdict(list)
    for noun, pattern in TRAIT_PATTERNS:
        for m in pattern.finditer(text):
            key = (m.group("owner").lower(), noun, m.group("value").lower())
            groups[key].append(m)
    for key in sorted(groups):
        owner, noun, value = key
        occurrences = sorted(groups[key], key=lambda m: m.start())
        if len(occurrences) < 2:
            continue
        alternatives = sorted(v for v in TRAIT_VALUES[noun] if not same_colour(v, value))
        for anchor, m in zip(occurrences, occurrences[1:]):
            replacement = rng.choice(alternatives)
            if m.group("value")[0].isupper():
                replacement = replacement.capitalize()
            yield Hit(
                "trait_flip",
                "fact_contradiction",
                m.start("value"),
                m.end("value"),
                m.group("value"),
                replacement,
                f"{owner} {noun} colour is stated as {value} earlier in the text",
                anchor=anchor.start(),
                long_range_ok=bool(NAME_OWNER.match(m.group("owner"))),
            )


def quoted_spans(text):
    """Character ranges covered by dialogue, so narration can be told from speech."""
    return [(m.start(), m.end()) for m in QUOTED_SPAN.finditer(text)]


def outside_quotes(spans, position):
    """True when position is narration rather than something a character says."""
    return not any(start <= position < end for start, end in spans)


# POV slip is deliberately NOT a rule here, and that is a measured decision rather
# than an omission. Injecting one by switching a narrated "he/she <verb>" to
# "I <verb>" in a third-person novel yields plenty - 17,258 sites across 14 novels,
# 180 items built - but the `narration_person` attack in evaluate.py detects 100% of
# them with five lines of regex and no model at all. The injected "I" is the only
# first-person narration token in an otherwise pure third-person passage, so it is a
# lexical outlier, not a contradiction that has to be reasoned about. Matching the
# clean controls on that statistic is not possible either: third-person novels sit at
# a 0.003 first-person narration ratio, so passages that already contain a narrated
# "I" barely exist.
#
# The useful conclusion for the project: a POV slip is surface-detectable and belongs
# in a linter, not in a benchmark meant to measure state tracking. The spec lists it
# as one of three candidate error types; this is the evidence for dropping it.

GLOBAL_RULES = (candidates_timeline, candidates_trait_flip)


def person_names(text):
    """Names with positive evidence of being a person: they speak or act, and take no article.

    A purely orthographic test keeps every capitalised token, which is how "The",
    "Master", "She", "Company" and "Mars" end up injected as characters and make an
    item nonsense rather than an error. Three filters remove them: a closed-class
    stoplist, evidence of speaking, and rejection of anything that regularly follows
    an article. Speaking is deliberately the only evidence accepted - acting admits
    planets and institutions, which act in prose all the time.

    A fourth filter requires a name to stand alone a fair share of the time, which
    is what separates a character from the fragment of a longer name: "Van" in "Van
    Helsing", "York" in "New York" and "Sir" in "Sir Leicester" never appear on
    their own, and injecting one of them produces gibberish.
    """
    evidence, determined, standalone = Counter(), Counter(), Counter()
    for pattern in (PERSON_AS_OBJECT, PERSON_AS_SUBJECT):
        for m in pattern.finditer(text):
            evidence[m.group("name")] += 1
    counts, matches = name_counts_with_matches(text)
    for m in matches:
        token = m.group(0)
        if token not in counts:
            continue
        before = text[max(0, m.start() - 20) : m.start()]
        if DETERMINED.search(before[-8:]):
            determined[token] += 1
        if not BLOCKED_BEFORE.search(before) and not BLOCKED_AFTER.match(text[m.end() : m.end() + 20]):
            standalone[token] += 1
    return Counter(
        {
            name: count
            for name, count in counts.items()
            if evidence[name] >= MIN_SPEECH_EVIDENCE
            and determined[name] <= MAX_ARTICLED_SHARE * count
            and standalone[name] >= MIN_STANDALONE_SHARE * count
        }
    )


def name_genders(text, names):
    """Guess each name's gender from the pronouns that follow its mentions.

    Swapping a man for a woman leaves the surrounding pronouns disagreeing, which
    makes the injected item detectable by grammar rather than by continuity
    reasoning. Requiring agreement keeps the benchmark measuring what it claims to.
    """
    scores = defaultdict(lambda: [0, 0])
    for m in NAME_RE.finditer(text):
        token = m.group(0)
        if token not in names:
            continue
        tail = text[m.end() : m.end() + PRONOUN_WINDOW]
        scores[token][0] += len(MALE_PRONOUNS.findall(tail))
        scores[token][1] += len(FEMALE_PRONOUNS.findall(tail))
    genders = {}
    for name, (male, female) in scores.items():
        if male >= 4 * max(1, female):
            genders[name] = "m"
        elif female >= 4 * max(1, male):
            genders[name] = "f"
    return genders


def alias_pairs(text, names):
    """Name pairs that appear adjacent, so they are probably one person.

    'Tom' and 'Sawyer' are the same character, so swapping one for the other is not
    a continuity error at all - it is the item silently labelled wrong. The same
    applies when the intruder is another name for someone already in the scene:
    inserting 'Hallward' where 'Basil' is standing three lines away is not an error.

    ponytail: adjacency is the whole test, so nickname pairs that never appear side
    by side ('Lizzy' for 'Elizabeth') still slip through. Add a name-cluster pass if
    the residual rate shows up in a spot-check.
    """
    pairs = set()
    for m in ADJACENT_NAMES.finditer(text):
        first, second = m.group("first"), m.group("second")
        if first in names and second in names:
            pairs.add(frozenset((first, second)))
    return pairs


def standalone_occurrences(text, name):
    """Occurrences of name that are not part of a longer name or an honorific phrase.

    'Mr. Copperfield' and 'Oliver Twist' must not be half-replaced: swapping one
    token of a two-token name produces gibberish, not a continuity error.
    """
    out = []
    for m in re.finditer(r"\b" + re.escape(name) + r"\b", text):
        if BLOCKED_BEFORE.search(text[max(0, m.start() - 20) : m.start()]):
            continue
        if BLOCKED_AFTER.match(text[m.end() : m.end() + 20]):
            continue
        out.append(m)
    return out


def rule_character_swap(text, rng, ctx):
    """Drop a character who is absent from the scene into it, in place of one who is present."""
    persons, genders = ctx["novel_persons"], ctx["novel_genders"]
    local, nearby = ctx["local_counts"], ctx["nearby_names"]
    present = sorted(
        name
        for name, count in local.items()
        if count >= MIN_LOCAL_MENTIONS and name in persons and name in genders
    )
    absent = sorted(
        name
        for name, count in persons.items()
        if count >= MIN_NOVEL_MENTIONS and name not in nearby and name in genders
    )
    if not present or not absent:
        return None
    victim = rng.choice(present)
    aliases = ctx["alias_pairs"]
    blocked = {victim} | {
        other for pair in aliases for other in pair if pair & nearby
    }
    candidates = [
        name
        for name in absent
        if genders[name] == genders[victim] and not any(frozenset((name, b)) in aliases for b in blocked)
    ]
    if not candidates:
        return None
    intruder = rng.choice(candidates)
    occurrences = list(re.finditer(r"\b" + re.escape(victim) + r"\b", text))
    if len(occurrences) < 3:
        return None
    swappable = [m for m in standalone_occurrences(text, victim) if m.start() > occurrences[0].start()]
    if not swappable:
        return None
    m = rng.choice(swappable)
    return Hit(
        "character_swap",
        "fact_contradiction",
        m.start(),
        m.end(),
        victim,
        intruder,
        f"{intruder} does not appear anywhere near this scene; the character present is {victim}",
        anchor=occurrences[0].start(),
        long_range_ok=True,
    )


def windows(paras, size):
    """Yield (start_index, paragraph_list) for non-overlapping windows of the given size."""
    for i in range(0, len(paras) - size + 1, size):
        yield i, paras[i : i + size]


def passage_around(full, paras, starts, hit, max_paragraphs, pad, max_chars):
    """Carve the smallest span of text that holds both the anchor and the edit.

    Returns (lo, hi, text, hit_with_local_offsets, scope), where scope is "passage"
    when the anchor and the edit fit inside max_paragraphs, and "long_range" when the
    fact is established chapters before it is contradicted. Long-range items are the
    ones that cannot be solved by reading a single passage, so they are kept rather
    than discarded.

    Returns None when the span exceeds max_chars, or when it would be long-range and
    the hit is not valid at that distance - scope policy lives here, in the one
    function that computes scope, rather than being re-derived by the caller.
    """
    first = paragraph_of(starts, hit.anchor)
    last = paragraph_of(starts, hit.start)
    if last < first:
        first, last = last, first
    if last - first + 1 <= max_paragraphs:
        scope = "passage"
        lo, hi = max(0, first - pad), min(len(paras) - 1, last + pad)
        while hi - lo + 1 > max_paragraphs:
            if hi - last > first - lo:
                hi -= 1
            else:
                lo += 1
    else:
        scope = "long_range"
        lo, hi = max(0, first - 1), min(len(paras) - 1, last + 1)
    if scope == "long_range" and not hit.long_range_ok:
        return None
    base, end = starts[lo], starts[hi] + len(paras[hi])
    if end - base > max_chars:
        return None
    local = replace(hit, start=hit.start - base, end=hit.end - base, anchor=hit.anchor - base)
    return lo, hi, full[base:end], local, scope


def split_for(novel_id, test_every=4):
    """Assign a whole novel to dev or test so no novel spans both splits."""
    digest = int(hashlib.sha1(novel_id.encode()).hexdigest(), 16)
    return "test" if digest % test_every == 0 else "dev"


def item_pair(novel_id, passage_id, split, text, hit, n_paragraphs, scope="passage"):
    """Build the (injected, clean control) pair of JSONL records for one passage."""
    injected = text[: hit.start] + hit.replacement + text[hit.end :]
    base = {
        "novel_id": novel_id,
        "pair_id": passage_id,
        "split": split,
        "n_paragraphs": n_paragraphs,
        "context_scope": scope,
    }
    err = {
        **base,
        "passage_id": f"{passage_id}-err",
        "text": injected,
        "error_type": hit.error_type,
        "injected_location": {
            "paragraph_index": text.count("\n\n", 0, hit.start),
            "char_start": hit.start,
            "char_end": hit.start + len(hit.replacement),
            "original": hit.original,
            "replacement": hit.replacement,
            "anchor_paragraph_index": text.count("\n\n", 0, hit.anchor),
        },
        "ground_truth": {"has_error": True, "rule": hit.rule, "description": hit.description},
    }
    clean = {
        **base,
        "passage_id": f"{passage_id}-clean",
        "text": text,
        "error_type": None,
        "injected_location": None,
        "ground_truth": {
            "has_error": False,
            "rule": None,
            "description": f"matched control for {hit.rule}",
        },
    }
    return err, clean


def novel_items(novel_id, paras, split, args):
    """Yield JSONL records for one novel, capped per rule to keep the corpus balanced."""
    full, starts = join_paragraphs(paras)
    block_counts = {}

    def block_names(index):
        """Name counts for one stride-aligned block, computed once and reused.

        Windows stride by a whole block, so each block is the local scene of one
        window and part of the surrounding context of its neighbours. Without this
        cache every block is scanned four times, which was the largest avoidable
        cost in the build.
        """
        if index not in block_counts:
            block = paras[index * args.window : (index + 1) * args.window]
            block_counts[index] = name_counts("\n\n".join(block))
        return block_counts[index]

    novel_persons = person_names(full)
    novel_genders = name_genders(full, novel_persons)
    novel_aliases = alias_pairs(full, novel_persons)
    pad = max(1, args.window // 3)
    used, per_rule = set(), Counter()

    for rule in GLOBAL_RULES:
        rng = random.Random(f"{args.seed}:{novel_id}:{rule.__name__}")
        for hit in rule(full, rng):
            if per_rule[hit.rule] >= args.cap_per_rule:
                break
            first, last = paragraph_of(starts, hit.anchor), paragraph_of(starts, hit.start)
            if first in used or last in used:
                continue
            carved = passage_around(full, paras, starts, hit, args.window, pad, args.max_chars)
            if carved is None:
                continue
            lo, hi, text, local, scope = carved
            passage_id = f"{novel_id}-p{lo:05d}"
            used.update((first, last))
            per_rule[hit.rule] += 1
            yield from item_pair(novel_id, passage_id, split, text, local, hi - lo + 1, scope)

    for index, window in windows(paras, args.window):
        if per_rule["character_swap"] >= args.cap_per_rule:
            break
        if any(index <= u < index + args.window for u in used):
            continue

        text = "\n\n".join(window)
        lo = max(0, index - args.window)
        hi = min(len(paras), index + 2 * args.window)
        nearby = set()
        for block in range(lo // args.window, -(-hi // args.window)):
            nearby.update(block_names(block))
        ctx = {
            "novel_persons": novel_persons,
            "novel_genders": novel_genders,
            "alias_pairs": novel_aliases,
            "local_counts": block_names(index // args.window),
            "nearby_names": nearby,
        }
        passage_id = f"{novel_id}-p{index:05d}"
        hit = rule_character_swap(text, random.Random(f"{args.seed}:{passage_id}"), ctx)
        if hit is None:
            continue
        used.update(range(index, index + args.window))
        per_rule["character_swap"] += 1
        yield from item_pair(novel_id, passage_id, split, text, hit, len(window))


def cmd_fetch(args):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetched, failed = [], []
    for book in read_books():
        book_id = book["id"].strip()
        dest = RAW_DIR / f"pg{book_id}.txt"
        if dest.exists() and dest.stat().st_size > 20000:
            fetched.append((book_id, gutenberg_title(dest.read_text(encoding="utf-8", errors="replace"))))
            continue
        text, last = None, None
        urls = (
            f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
            f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8",
        )
        for url in urls:
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=60) as response:
                    text = response.read().decode("utf-8", errors="replace")
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
                last = exc
        if text is None or len(text) < 20000:
            failed.append((book_id, book.get("title", ""), type(last).__name__ if last else "too short"))
            continue
        dest.write_text(text, encoding="utf-8")
        fetched.append((book_id, gutenberg_title(text)))
        print(f"fetched pg{book_id}  {fetched[-1][1]}")
        time.sleep(args.delay)
    FETCHED_TSV.parent.mkdir(parents=True, exist_ok=True)
    with FETCHED_TSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["novel_id", "detected_title"])
        for book_id, title in fetched:
            writer.writerow([f"pg{book_id}", title])
    print(f"\n{len(fetched)} available, {len(failed)} failed")
    for book_id, title, reason in failed:
        print(f"  FAILED pg{book_id} {title!r}: {reason}")
    return 0


def cmd_build(args):
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rules_seen, novels_used, items = Counter(), 0, 0
    with out_path.open("w", encoding="utf-8") as fh:
        for path in sorted(RAW_DIR.glob("pg*.txt")):
            novel_id = path.stem
            paras = paragraphs(strip_boilerplate(path.read_text(encoding="utf-8", errors="replace")))
            if len(paras) < args.window * 3:
                print(f"skip {novel_id}: only {len(paras)} usable paragraphs", file=sys.stderr)
                continue
            split = split_for(novel_id)
            local = Counter()
            for record in novel_items(novel_id, paras, split, args):
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                items += 1
                if record["ground_truth"]["has_error"]:
                    local[record["ground_truth"]["rule"]] += 1
            if local:
                novels_used += 1
                rules_seen.update(local)
                print(f"{novel_id} [{split}]: {dict(local.most_common())}")
    print(f"\nwrote {items} items ({items // 2} matched pairs) from {novels_used} novels to {out_path}")
    for rule, count in rules_seen.most_common():
        print(f"  {rule}: {count}")
    return 0


def cmd_stats(args):
    by_rule, by_split, by_novel, by_scope = Counter(), Counter(), Counter(), Counter()
    lengths = defaultdict(list)
    with Path(args.corpus).open(encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            by_rule[item["ground_truth"]["rule"] or "clean_control"] += 1
            by_split[item["split"]] += 1
            by_novel[item["novel_id"]] += 1
            by_scope[item["context_scope"]] += 1
            lengths[item["context_scope"]].append(len(item["text"]))
    print(f"items: {sum(by_rule.values())}   novels: {len(by_novel)}")
    print("by label:", dict(by_rule.most_common()))
    print("by split:", dict(by_split))
    print("by scope:", dict(by_scope))
    for scope, values in sorted(lengths.items()):
        print(f"  {scope}: mean {sum(values) // len(values)} chars, max {max(values)}")
    return 0


def _excerpt(text, centre, half):
    """Return a window of text around centre, with ellipses where it was cut."""
    lo, hi = max(0, centre - half), min(len(text), centre + half)
    prefix = "..." if lo else ""
    suffix = "..." if hi < len(text) else ""
    return prefix + text[lo:hi] + suffix


def validate_corpus(rows):
    """Return a list of problems found in a built corpus, empty if it is sound."""
    problems = []
    by_pair = defaultdict(list)
    seen_ids, seen_text = set(), {}
    for row in rows:
        if row["passage_id"] in seen_ids:
            problems.append(f"duplicate passage_id {row['passage_id']}")
        seen_ids.add(row["passage_id"])
        if row["text"] in seen_text:
            problems.append(f"{row['passage_id']} duplicates the text of {seen_text[row['text']]}")
        seen_text[row["text"]] = row["passage_id"]
        by_pair[row["pair_id"]].append(row)
        loc = row["injected_location"]
        if row["ground_truth"]["has_error"] != (loc is not None):
            problems.append(f"{row['passage_id']} label and injected_location disagree")
            continue
        if loc is None:
            continue
        if row["text"][loc["char_start"] : loc["char_end"]] != loc["replacement"]:
            problems.append(f"{row['passage_id']} text does not hold the replacement at char_start")
    for pair_id, group in by_pair.items():
        if len(group) != 2:
            problems.append(f"pair {pair_id} has {len(group)} items, expected an injected/clean pair")
            continue
        errored = [r for r in group if r["ground_truth"]["has_error"]]
        clean = [r for r in group if not r["ground_truth"]["has_error"]]
        if len(errored) != 1 or len(clean) != 1:
            problems.append(f"pair {pair_id} is not one injected item and one control")
            continue
        if errored[0]["text"] == clean[0]["text"]:
            problems.append(f"pair {pair_id} injected item is identical to its control")
        if errored[0]["novel_id"] != clean[0]["novel_id"] or errored[0]["split"] != clean[0]["split"]:
            problems.append(f"pair {pair_id} spans two novels or two splits")
    return problems


def cmd_validate(args):
    rows = [json.loads(line) for line in Path(args.corpus).open(encoding="utf-8")]
    problems = validate_corpus(rows)
    for problem in problems[: args.limit]:
        print(f"PROBLEM {problem}")
    print(f"{len(rows)} items checked, {len(problems)} problem(s)")
    return 1 if problems else 0


def write_jsonl(path, records):
    """Write records as JSON lines, keeping non-ASCII text as itself."""
    with Path(path).open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def opaque_id(seed, passage_id):
    """Stable, label-free public id for a passage.

    The internal passage_id ends in -err or -clean, which would publish the answer
    for every test item; the mapping back lives only in the held-back labels file.
    """
    return hashlib.sha1(f"{seed}:{passage_id}".encode()).hexdigest()[:16]


def one_half_per_pair(rows, seed):
    """Keep one half of each matched pair, chosen by hash of the pair id.

    Publishing both halves of a pair is an answer key, and not a weak one: the two
    texts are near-identical, so clustering them and diffing hands an attacker the
    exact token to test, and the injected half is the one where that token appears
    once while its counterpart appears often. Measured at J +0.699 against this
    corpus, against the J +0.066 an attacker manages without the pair. Shipping one
    half per pair removes the attack rather than arguing about its strength.

    The cost is exact per-pair matching on the released test split; dev keeps both
    halves, because its job is for humans to read. The property that matters - a
    false-positive rate measured on the same passage pool as the recall - survives in
    expectation, since each pair contributes one half drawn from that pool.
    """
    by_pair = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)
    kept = []
    for pair_id, group in sorted(by_pair.items()):
        group.sort(key=lambda row: row["passage_id"])
        digest = int(hashlib.sha1(f"{seed}:half:{pair_id}".encode()).hexdigest(), 16)
        kept.append(group[digest % len(group)])
    return kept


def cmd_export(args):
    """Write the release files: labeled dev, unlabeled test passages, held-back labels.

    Three things protect the test split, because the matched-pair design is itself an
    answer key: labels are held back, ids are opaque so the internal -err / -clean
    suffix does not publish the answer, and only one half of each pair ships at all
    (see one_half_per_pair for the measurement that forced the last one). The dev
    split ships whole and fully labeled, because its purpose is for people to read.
    """
    rows = [json.loads(line) for line in Path(args.corpus).open(encoding="utf-8")]
    problems = validate_corpus(rows)
    if problems and not args.force:
        print(f"refusing to export: {len(problems)} validation problem(s); run `validate`")
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = [row for row in rows if row["split"] == "dev"]
    test = one_half_per_pair([row for row in rows if row["split"] == "test"], args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(dev)
    rng.shuffle(test)
    public_fields = ("novel_id", "context_scope", "n_paragraphs", "text")
    write_jsonl(out / "dev.jsonl", dev)
    write_jsonl(
        out / "test_passages.jsonl",
        (
            {
                "passage_id": opaque_id(args.seed, row["passage_id"]),
                **{key: row[key] for key in public_fields},
            }
            for row in test
        ),
    )
    write_jsonl(
        out / "test_labels.jsonl",
        (
            {
                "passage_id": opaque_id(args.seed, row["passage_id"]),
                "source_passage_id": row["passage_id"],
                "pair_id": row["pair_id"],
                "error_type": row["error_type"],
                "injected_location": row["injected_location"],
                "ground_truth": row["ground_truth"],
            }
            for row in test
        ),
    )
    print(f"dev.jsonl            {len(dev):4} labeled items ({len(dev) // 2} pairs)")
    print(f"test_passages.jsonl  {len(test):4} passages, one half per pair, opaque ids")
    print(f"test_labels.jsonl    {len(test):4} labels - HOLD BACK, do not publish")
    print(f"\nwrote to {out}")
    return 0


def cmd_sample(args):
    """Print injected items with the edit marked, so a human can check the labels."""
    items = []
    with Path(args.corpus).open(encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            if not item["ground_truth"]["has_error"]:
                continue
            if args.rule and item["ground_truth"]["rule"] != args.rule:
                continue
            items.append(item)
    half = max(200, args.chars // 2)
    for item in random.Random(args.seed).sample(items, min(args.n, len(items))):
        loc = item["injected_location"]
        text = item["text"]
        marked = (
            text[: loc["char_start"]]
            + f">>> {loc['replacement']} (was {loc['original']}) <<<"
            + text[loc["char_end"] :]
        )
        print("=" * 100)
        print(
            f"{item['passage_id']}  rule={item['ground_truth']['rule']}  scope={item['context_scope']}  "
            f"chars={len(text)}\n{item['ground_truth']['description']}"
        )
        print("-" * 100)
        anchor = text.find(loc["original"])
        print("ANCHOR CONTEXT:")
        print(_excerpt(marked, max(0, anchor), half))
        print("\nEDIT CONTEXT:")
        print(_excerpt(marked, loc["char_start"], half))
        print()
    print(f"{len(items)} injected items matched")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download novels listed in books.tsv")
    fetch.add_argument("--delay", type=float, default=1.0, help="seconds between downloads")
    fetch.set_defaults(func=cmd_fetch)

    build = sub.add_parser("build", help="inject errors and emit the JSONL corpus")
    build.add_argument("--out", default=str(DEFAULT_CORPUS))
    build.add_argument("--window", type=int, default=12, help="maximum paragraphs per passage")
    build.add_argument("--cap-per-rule", type=int, default=12, help="injected passages per rule per novel")
    build.add_argument("--max-chars", type=int, default=400000, help="skip candidates whose span exceeds this")
    build.add_argument("--seed", default="continuity-bench-v0")
    build.set_defaults(func=cmd_build)

    stats = sub.add_parser("stats", help="summarise a built corpus")
    stats.add_argument("corpus", nargs="?", default=str(DEFAULT_CORPUS))
    stats.set_defaults(func=cmd_stats)

    validate = sub.add_parser("validate", help="check a built corpus for label and pairing faults")
    validate.add_argument("corpus", nargs="?", default=str(DEFAULT_CORPUS))
    validate.add_argument("--limit", type=int, default=20, help="problems to print")
    validate.set_defaults(func=cmd_validate)

    export = sub.add_parser("export", help="write the publishable release files")
    export.add_argument("corpus", nargs="?", default=str(DEFAULT_CORPUS))
    export.add_argument("--out", default=str(ROOT / "release"))
    export.add_argument("--seed", default="release-v0")
    export.add_argument("--force", action="store_true", help="export even if validation fails")
    export.set_defaults(func=cmd_export)

    sample = sub.add_parser("sample", help="print injected items with the edit marked")
    sample.add_argument("corpus", nargs="?", default=str(DEFAULT_CORPUS))
    sample.add_argument("--rule", help="only this rule")
    sample.add_argument("-n", type=int, default=5)
    sample.add_argument("--chars", type=int, default=2500, help="truncate each passage")
    sample.add_argument("--seed", default="sample")
    sample.set_defaults(func=cmd_sample)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
