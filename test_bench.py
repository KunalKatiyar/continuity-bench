"""Self-checks for bench.py. Run directly (python test_bench.py) or under pytest."""

import random

import bench


def _rng():
    return random.Random("test")


def test_strip_boilerplate_drops_header_and_footer():
    raw = (
        "licence blurb\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n"
        "real prose\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n"
        "trailing junk\n"
    )
    assert bench.strip_boilerplate(raw).strip() == "real prose"


def test_paragraphs_normalises_and_filters():
    text = (
        "CHAPTER IV\n\n"
        "short\n\n"
        "A paragraph long enough to survive the minimum length filter, written\nacross two lines\nof source text.\n"
    )
    paras = bench.paragraphs(text)
    assert len(paras) == 1
    assert "\n" not in paras[0]
    assert paras[0].startswith("A paragraph long enough")


def _first(gen):
    return next(iter(gen), None)


def test_timeline_rule_breaks_a_consistent_day_sequence():
    text = (
        "He reached the inn on Tuesday evening, tired and wet through from the long ride over the moor. "
        "The next morning, a Wednesday, he set out again before the household had stirred."
    )
    hit = _first(bench.candidates_timeline(text, _rng()))
    assert hit is not None and hit.rule == "timeline_weekday"
    assert hit.original == "Wednesday" and hit.replacement == "Friday"
    assert text[hit.start : hit.end] == "Wednesday"


def test_timeline_rule_ignores_an_already_inconsistent_sequence():
    text = (
        "He reached the inn on Tuesday evening, tired and wet through from the long ride over the moor. "
        "The next morning, a Saturday, he set out again before the household had stirred."
    )
    assert _first(bench.candidates_timeline(text, _rng())) is None


def test_trait_rule_flips_the_second_mention_only():
    text = (
        "She was a small woman with her dark hair pinned back and a manner that discouraged questions. "
        "Later, when the lamp was lit, her dark hair looked almost black against the papered wall."
    )
    hit = _first(bench.candidates_trait_flip(text, _rng()))
    assert hit is not None and hit.error_type == "fact_contradiction"
    assert hit.original.lower() == "dark" and hit.replacement.lower() != "dark"
    assert hit.start > text.index("her dark hair")


def test_trait_rule_needs_two_mentions():
    text = "She was a small woman with her dark hair pinned back and a manner that discouraged questions."
    assert _first(bench.candidates_trait_flip(text, _rng())) is None


def test_character_swap_inserts_an_absent_character():
    text = (
        "Margaret crossed the yard without looking back at the house she had just left behind her. "
        "The gate stuck, as it always stuck, and Margaret shouldered it open with the patience of habit. "
        "Whatever Margaret had expected of the morning, it was not this thin and colourless rain."
    )
    ctx = {
        "novel_persons": bench.Counter({"Margaret": 400, "Hollis": 90}),
        "novel_genders": {"Margaret": "f", "Hollis": "f"},
        "alias_pairs": set(),
        "local_counts": bench.name_counts(text),
        "nearby_names": set(bench.name_counts(text)),
    }
    hit = bench.rule_character_swap(text, _rng(), ctx)
    assert hit is not None
    assert hit.original == "Margaret" and hit.replacement == "Hollis"
    assert hit.start > text.index("Margaret")


def test_character_swap_skips_characters_present_nearby():
    text = (
        "Margaret crossed the yard without looking back at the house she had just left behind her. "
        "The gate stuck, as it always stuck, and Margaret shouldered it open with the patience of habit. "
        "Whatever Margaret had expected of the morning, it was not this thin and colourless rain."
    )
    ctx = {
        "novel_persons": bench.Counter({"Margaret": 400, "Hollis": 90}),
        "novel_genders": {"Margaret": "f", "Hollis": "f"},
        "alias_pairs": set(),
        "local_counts": bench.name_counts(text),
        "nearby_names": set(
            bench.name_counts(text + " A page later, Hollis came up the lane with the dog at his heel.")
        ),
    }
    assert bench.rule_character_swap(text, _rng(), ctx) is None


def test_injection_is_a_single_contiguous_span_edit():
    text = (
        "He reached the inn on Tuesday evening, tired and wet through from the long ride over the moor. "
        "The next morning, a Wednesday, he set out again before the household had stirred."
    )
    hit = _first(bench.candidates_timeline(text, _rng()))
    injected = text[: hit.start] + hit.replacement + text[hit.end :]
    assert injected != text
    assert injected[: hit.start] == text[: hit.start]
    assert injected[hit.start : hit.start + len(hit.replacement)] == hit.replacement
    assert injected[hit.start + len(hit.replacement) :] == text[hit.end :]


def test_name_counts_keeps_honorific_names_and_drops_sentence_openers():
    text = "It rained. Mrs. Gaskell said nothing to Mr. Thornton, and Thornton said nothing at all to her."
    counts = bench.name_counts(text)
    assert counts["Gaskell"] == 1
    assert counts["Thornton"] == 2
    assert "Chapter" not in counts


def test_split_is_stable_per_novel():
    assert bench.split_for("pg1342") == bench.split_for("pg1342")
    assert bench.split_for("pg1342") in ("dev", "test")


def test_trait_rule_ignores_different_owners():
    text = (
        "She was a small woman with her dark hair pinned back and a manner that discouraged questions. "
        "Later, when the lamp was lit, his dark hair looked almost black against the papered wall."
    )
    assert _first(bench.candidates_trait_flip(text, _rng())) is None


def test_paragraph_offsets_round_trip():
    paras = ["a" * 100, "b" * 120, "c" * 90]
    full, starts = bench.join_paragraphs(paras)
    assert full.count("\n\n") == 2
    for i, start in enumerate(starts):
        assert bench.paragraph_of(starts, start) == i
        assert bench.paragraph_of(starts, start + 5) == i
        assert full[start : start + len(paras[i])] == paras[i]


def test_passage_around_keeps_offsets_and_fits_the_window():
    paras = [f"paragraph {i} " + "filler text to clear the minimum length bar. " * 3 for i in range(40)]
    full, starts = bench.join_paragraphs(paras)
    target = full.index("paragraph 20")
    hit = bench.Hit("r", "t", target, target + len("paragraph"), "paragraph", "PARA", "d", anchor=starts[18])
    lo, hi, text, local, scope = bench.passage_around(
        full, paras, starts, hit, max_paragraphs=12, pad=4, max_chars=10**6
    )
    assert scope == "passage"
    assert hi - lo + 1 <= 12
    assert lo <= 18 and hi >= 20
    assert text[local.start : local.end] == "paragraph"
    assert text == full[starts[lo] : starts[hi] + len(paras[hi])]


def test_passage_around_keeps_a_distant_anchor_as_long_range():
    paras = [f"paragraph {i} " + "filler text to clear the minimum length bar. " * 3 for i in range(40)]
    full, starts = bench.join_paragraphs(paras)
    target = full.index("paragraph 30")
    hit = bench.Hit(
        "r", "t", target, target + 9, "paragraph", "PARA", "d", anchor=starts[2], long_range_ok=True
    )
    lo, hi, text, local, scope = bench.passage_around(
        full, paras, starts, hit, max_paragraphs=12, pad=4, max_chars=10**6
    )
    assert scope == "long_range"
    assert lo <= 2 and hi >= 30
    assert text[local.start : local.end] == "paragraph"
    assert text[local.anchor : local.anchor + 11] == "paragraph 2"


def test_passage_around_rejects_a_span_over_the_char_cap():
    paras = [f"paragraph {i} " + "filler text to clear the minimum length bar. " * 3 for i in range(40)]
    full, starts = bench.join_paragraphs(paras)
    target = full.index("paragraph 30")
    hit = bench.Hit("r", "t", target, target + 9, "paragraph", "PARA", "d", anchor=starts[2])
    assert bench.passage_around(full, paras, starts, hit, max_paragraphs=12, pad=4, max_chars=500) is None


def test_item_pair_labels_and_span_agree():
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen."
    hit = bench.Hit("r", "fact_contradiction", 4, 7, "two", "ninety", "d", anchor=0)
    err, clean = bench.item_pair("pgX", "pgX-p00001", "dev", text, hit, 12)
    assert err["context_scope"] == "passage"
    loc = err["injected_location"]
    assert err["text"][loc["char_start"] : loc["char_end"]] == "ninety"
    assert err["ground_truth"]["has_error"] is True
    assert clean["text"] == text and clean["error_type"] is None
    assert err["pair_id"] == clean["pair_id"]


def test_trait_rule_accepts_a_curly_apostrophe_possessive():
    text = (
        "It was Anna\u2019s golden hair that everyone in the room remembered afterwards, and little else. "
        "By the time the lamps were lit again, Anna\u2019s golden hair had come loose beneath her hat."
    )
    hit = _first(bench.candidates_trait_flip(text, _rng()))
    assert hit is not None
    assert hit.long_range_ok is True


def test_trait_rule_marks_pronoun_owners_as_passage_only():
    text = (
        "She was a small woman with her dark hair pinned back and a manner that discouraged questions. "
        "Later, when the lamp was lit, her dark hair looked almost black against the papered wall."
    )
    hit = _first(bench.candidates_trait_flip(text, _rng()))
    assert hit is not None
    assert hit.long_range_ok is False


def test_person_names_rejects_titles_and_non_people():
    text = (
        "The evening was cold, and Master Copperfield said nothing as the London road lay empty. "
        "Then Copperfield replied at last, and Copperfield said he had seen enough of The Rookery. "
        "The London road was long. The Rookery was shut. The Master waited, and London waited too."
    )
    persons = bench.person_names(text)
    assert "Copperfield" in persons
    for token in ("The", "Master", "London", "Rookery"):
        assert token not in persons, token


def test_standalone_occurrences_skips_honorific_and_full_name_positions():
    text = (
        "Mr. Copperfield bowed to the room. Copperfield had nothing to add to that. "
        "Later, David Copperfield wrote it all down exactly as it had happened that night."
    )
    spans = bench.standalone_occurrences(text, "Copperfield")
    assert len(spans) == 1
    assert text[spans[0].start() - 2 : spans[0].start()] == ". "


def test_character_swap_will_not_split_a_full_name():
    text = (
        "Mr. Copperfield came in from the rain and stood a long while by the shuttered window. "
        "David Copperfield had written of that evening more than once, and never the same way twice. "
        "Nobody spoke to Mr. Copperfield, and Mr. Copperfield spoke to nobody at all that evening."
    )
    ctx = {
        "novel_persons": bench.Counter({"Copperfield": 400, "Traddles": 90}),
        "novel_genders": {"Copperfield": "m", "Traddles": "m"},
        "alias_pairs": set(),
        "local_counts": bench.name_counts(text),
        "nearby_names": set(bench.name_counts(text)),
    }
    assert bench.rule_character_swap(text, _rng(), ctx) is None


def test_person_names_rejects_pronouns_and_articled_nouns():
    text = (
        "She said nothing at all, and the Company said less, but Copperfield said the rest of it. "
        "She said it twice over, the Company said it thrice, and Copperfield replied to neither. "
        "The Company was patient. She was not. In the end Copperfield answered and walked out."
    )
    persons = bench.person_names(text)
    assert "Copperfield" in persons
    for token in ("She", "Company", "The"):
        assert token not in persons, token


def test_name_genders_reads_following_pronouns():
    text = (
        "Edmund put down his hat and rubbed his hands, for he was cold and he had walked far. "
        "Fanny took her shawl from her shoulders, because she was warm and she had sat all day. "
        "Edmund said he would wait. Fanny said she would not."
    )
    genders = bench.name_genders(text, {"Edmund", "Fanny"})
    assert genders["Edmund"] == "m"
    assert genders["Fanny"] == "f"


def test_character_swap_refuses_a_cross_gender_intruder():
    text = (
        "Oliver was awakened in the morning by a loud kicking at the outside of the shop door. "
        "Before Oliver could huddle on his clothes, the kicking was repeated in an angry manner. "
        "Whatever Oliver had expected of the morning, it was not this thin and colourless rain."
    )
    ctx = {
        "novel_persons": bench.Counter({"Oliver": 400, "Nancy": 90}),
        "novel_genders": {"Oliver": "m", "Nancy": "f"},
        "alias_pairs": set(),
        "local_counts": bench.name_counts(text),
        "nearby_names": set(bench.name_counts(text)),
    }
    assert bench.rule_character_swap(text, _rng(), ctx) is None


def test_alias_pairs_links_a_first_name_to_its_surname():
    text = "Tom Sawyer went down the lane. Tom said nothing. Sawyer said nothing either, for once."
    pairs = bench.alias_pairs(text, {"Tom", "Sawyer"})
    assert frozenset(("Tom", "Sawyer")) in pairs


def test_character_swap_refuses_the_victims_own_other_name():
    text = (
        "Bringing water from the town pump had always been hateful work in Tom's eyes before now. "
        "Tom sat down on the tree box, discouraged, and considered the whitewashed fence again. "
        "Whatever Tom had expected of the morning, it was not this thin and colourless rain."
    )
    ctx = {
        "novel_persons": bench.Counter({"Tom": 400, "Sawyer": 90}),
        "novel_genders": {"Tom": "m", "Sawyer": "m"},
        "alias_pairs": {frozenset(("Tom", "Sawyer"))},
        "local_counts": bench.name_counts(text),
        "nearby_names": set(bench.name_counts(text)),
    }
    assert bench.rule_character_swap(text, _rng(), ctx) is None


def test_person_names_rejects_a_planet_that_acts_but_never_speaks():
    text = (
        "The vegetable kingdom on Mars, instead of green, is of a vivid blood-red tint all over. "
        "In Mars the seeds were different, and Mars had been dying through a long slow age. "
        "But Ogilvy said as much himself, and Ogilvy replied again to me the following morning."
    )
    persons = bench.person_names(text)
    assert "Ogilvy" in persons
    assert "Mars" not in persons


def test_character_swap_refuses_an_alias_of_someone_in_the_scene():
    text = (
        "The elder man laughed and said the only horrible thing in the world was Dorian's ennui. "
        "To that Basil said nothing, and Dorian said nothing either, and the room stayed quiet. "
        "Whatever Dorian had expected of the evening, it was not this thin and colourless talk."
    )
    ctx = {
        "novel_persons": bench.Counter({"Dorian": 400, "Basil": 200, "Hallward": 90}),
        "novel_genders": {"Dorian": "m", "Basil": "m", "Hallward": "m"},
        "alias_pairs": {frozenset(("Basil", "Hallward"))},
        "local_counts": bench.name_counts(text),
        "nearby_names": set(bench.name_counts(text)),
    }
    assert bench.rule_character_swap(text, _rng(), ctx) is None


def test_honorifics_are_never_treated_as_names():
    text = "It rained hard. Then said Mrs. March, patting her pocket, that Mrs. March had a treasure."
    assert "Mrs" not in bench.person_names(text)
    assert "Mrs" in bench.NOT_NAMES


def test_standalone_occurrences_skips_an_honorific_period_name():
    text = "He bowed. Then said Mrs. March to the room, and Mrs waited, and Mrs. March waited too."
    spans = bench.standalone_occurrences(text, "Mrs")
    assert [m.start() for m in spans] == [text.index("Mrs waited")]


def _pair(pair_id="p1", **over):
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen."
    hit = bench.Hit("r", "fact_contradiction", 4, 7, "two", "ninety", "d", anchor=0)
    err, clean = bench.item_pair("pgX", pair_id, "dev", text, hit, 12)
    err.update(over)
    return [err, clean]


def test_validate_accepts_a_sound_pair():
    assert bench.validate_corpus(_pair()) == []


def test_validate_catches_a_label_that_lost_its_location():
    rows = _pair(injected_location=None)
    assert any("disagree" in problem for problem in bench.validate_corpus(rows))


def test_validate_catches_a_span_that_no_longer_holds_the_replacement():
    rows = _pair()
    rows[0]["text"] = rows[0]["text"].replace("ninety", "eleven")
    assert any("replacement" in problem for problem in bench.validate_corpus(rows))


def test_validate_catches_a_control_identical_to_its_injected_item():
    rows = _pair()
    rows[0]["text"] = rows[1]["text"]
    problems = bench.validate_corpus(rows)
    assert any("identical" in problem for problem in problems)


def test_validate_catches_an_unpaired_item():
    rows = _pair()[:1]
    assert any("expected an injected/clean pair" in problem for problem in bench.validate_corpus(rows))


def test_person_names_rejects_a_fragment_of_a_longer_name():
    text = (
        "In the end Van Helsing said it plainly, and Van Helsing said it twice over that night. "
        "We had come from New York, and New York was far behind us now, and Van Helsing knew it. "
        "But Seward said nothing of the kind, and Seward replied only when he was asked to."
    )
    persons = bench.person_names(text)
    assert "Seward" in persons
    for fragment in ("Van", "York"):
        assert fragment not in persons, fragment


def test_opaque_id_hides_the_err_and_clean_suffix():
    err = bench.opaque_id("s", "pg1-p00001-err")
    clean = bench.opaque_id("s", "pg1-p00001-clean")
    for value in (err, clean):
        assert "err" not in value and "clean" not in value
        assert len(value) == 16
    assert err != clean
    assert err == bench.opaque_id("s", "pg1-p00001-err")
    assert err != bench.opaque_id("other-seed", "pg1-p00001-err")


def test_passage_around_refuses_a_long_range_hit_that_is_not_valid_at_distance():
    paras = [f"paragraph {i} " + "filler text to clear the minimum length bar. " * 3 for i in range(40)]
    full, starts = bench.join_paragraphs(paras)
    target = full.index("paragraph 30")
    hit = bench.Hit(
        "r", "t", target, target + 9, "paragraph", "PARA", "d", anchor=starts[2], long_range_ok=False
    )
    assert bench.passage_around(full, paras, starts, hit, 12, 4, 10**6) is None


def test_passage_around_preserves_fields_it_does_not_rewrite():
    paras = [f"paragraph {i} " + "filler text to clear the minimum length bar. " * 3 for i in range(40)]
    full, starts = bench.join_paragraphs(paras)
    target = full.index("paragraph 20")
    hit = bench.Hit(
        "myrule", "mytype", target, target + 9, "paragraph", "PARA", "why", anchor=starts[18],
        long_range_ok=True,
    )
    _, _, _, local, _ = bench.passage_around(full, paras, starts, hit, 12, 4, 10**6)
    assert (local.rule, local.error_type, local.original, local.replacement, local.description) == (
        "myrule",
        "mytype",
        "paragraph",
        "PARA",
        "why",
    )
    assert local.long_range_ok is True


def test_the_filters_reject_world_knowledge_without_any_stoplist_help():
    """Places, nationalities and nobility must not need a stoplist entry.

    This is the test that keeps NOT_NAMES from growing by guesswork: every word here
    is absent from the stoplist, so if person_names starts accepting one, the fix is a
    filter, not a new word.
    """
    for token, text in (
        (
            "London",
            "We came from London, and the London road was long, and London had been kind. "
            "But Seward said as much himself, and Seward replied when he was asked to.",
        ),
        (
            "Mars",
            "The vegetable kingdom on Mars is blood-red, and in Mars the seeds were other. "
            "But Ogilvy said as much himself, and Ogilvy replied again the following morning.",
        ),
        (
            "Company",
            "The Company had waited, and the Company was patient, and the Company said so. "
            "But Marlow said as much himself, and Marlow replied when he was asked to.",
        ),
    ):
        assert token not in bench.NOT_NAMES, f"{token} should not need a stoplist entry"
        assert token not in bench.person_names(text), f"filters failed to reject {token}"


def test_every_stoplist_entry_is_a_closed_class_word():
    """No geography, nationalities or holidays - the filters handle those."""
    for token in ("England", "English", "France", "French", "German", "London", "Christmas"):
        assert token not in bench.NOT_NAMES, token


if __name__ == "__main__":
    import _selftest

    raise SystemExit(1 if _selftest.run(vars().copy()) else 0)
