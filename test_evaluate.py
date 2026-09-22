"""Self-checks for evaluate.py. Run directly (python test_evaluate.py) or under pytest."""

import evaluate


def _item(pair_id, has_error, rule=None, paragraph=3, novel="pgX", text="a\n\nb\n\nc\n\nd"):
    return {
        "novel_id": novel,
        "pair_id": pair_id,
        "passage_id": f"{pair_id}-{'err' if has_error else 'clean'}",
        "split": "dev",
        "context_scope": "passage",
        "n_paragraphs": 4,
        "text": text,
        "error_type": "fact_contradiction" if has_error else None,
        "injected_location": (
            {"paragraph_index": paragraph, "char_start": 0, "char_end": 1, "original": "x", "replacement": "y"}
            if has_error
            else None
        ),
        "ground_truth": {"has_error": has_error, "rule": rule, "description": ""},
    }


def _fixed(has_error, paragraph=None):
    return lambda item: evaluate.Prediction(has_error=has_error, paragraph_index=paragraph)


def test_perfect_predictor_scores_one():
    items = [_item("p1", True, "r"), _item("p1", False)]
    scores = evaluate.score(items, lambda i: evaluate.Prediction(i["ground_truth"]["has_error"], 3))
    assert scores.precision == 1.0 and scores.recall == 1.0
    assert scores.f1 == 1.0 and scores.youden_j == 1.0
    assert scores.localization == 1.0
    assert scores.false_positive_rate == 0.0


def test_j_is_reported_and_balanced_accuracy_is_not_since_it_is_j_rescaled():
    """balanced_accuracy was (1 + J) / 2 exactly, so it carried no extra information."""
    items = [_item("p1", True, "r"), _item("p1", False)]
    summary = evaluate.score(items, _fixed(True, 3)).summary()
    assert "youden_j" in summary
    assert "balanced_accuracy" not in summary


def test_flag_everything_scores_f1_two_thirds_and_j_zero():
    items = []
    for n in range(10):
        items += [_item(f"p{n}", True, "r"), _item(f"p{n}", False)]
    scores = evaluate.score(items, _fixed(True, 0))
    assert scores.recall == 1.0
    assert scores.false_positive_rate == 1.0
    assert round(scores.f1, 4) == 0.6667
    assert scores.youden_j == 0.0


def test_flag_nothing_scores_zero_everywhere():
    items = [_item("p1", True, "r"), _item("p1", False)]
    scores = evaluate.score(items, _fixed(False))
    assert scores.recall == 0.0 and scores.precision == 0.0 and scores.f1 == 0.0
    assert scores.youden_j == 0.0
    assert scores.true_negative == 1 and scores.false_negative == 1


def test_counts_land_in_the_right_cells():
    items = [_item("p1", True, "r"), _item("p1", False), _item("p2", True, "r"), _item("p2", False)]
    predictions = iter([True, True, False, False])
    scores = evaluate.score(items, lambda i: evaluate.Prediction(next(predictions), 3))
    assert scores.true_positive == 1
    assert scores.false_positive == 1
    assert scores.false_negative == 1
    assert scores.true_negative == 1
    assert scores.precision == 0.5 and scores.recall == 0.5


def test_localization_needs_the_right_paragraph():
    items = [_item("p1", True, "r", paragraph=3)]
    assert evaluate.score(items, _fixed(True, 3)).localization == 1.0
    assert evaluate.score(items, _fixed(True, 2)).localization == 0.0


def test_localization_is_not_credited_without_detection():
    items = [_item("p1", True, "r", paragraph=3)]
    scores = evaluate.score(items, _fixed(False, 3))
    assert scores.localization == 0.0


def test_recall_is_broken_out_by_rule_and_novel():
    items = [
        _item("p1", True, "character_swap", novel="pgA"),
        _item("p2", True, "trait_flip", novel="pgB"),
    ]
    flags = iter([True, False])
    scores = evaluate.score(items, lambda i: evaluate.Prediction(next(flags), 3))
    summary = scores.summary()
    assert summary["recall_by_rule"] == {"character_swap": 1.0, "trait_flip": 0.0}
    assert summary["recall_by_novel"] == {"pgA": 1.0, "pgB": 0.0}


def test_cost_and_tokens_accumulate():
    items = [_item("p1", True, "r"), _item("p1", False)]
    scores = evaluate.score(
        items, lambda i: evaluate.Prediction(True, 3, cost_usd=0.01, input_tokens=100, output_tokens=5)
    )
    assert round(scores.cost_usd, 6) == 0.02
    assert scores.input_tokens == 200 and scores.output_tokens == 10


def test_fit_hook_sees_every_item_before_scoring():
    seen = {}

    class Stateful:
        needs_key = False
        cost_class = "free"

        def fit(self, items):
            seen["n"] = len(items)

        def __call__(self, item):
            return evaluate.Prediction(False)

    items = [_item("p1", True, "r"), _item("p1", False)]
    evaluate.score(items, Stateful())
    assert seen["n"] == 2


def test_corpus_prior_finds_a_planted_locally_rare_globally_common_name():
    common = " ".join(f"Margaret said thing {n}." for n in range(30))
    passage = (
        "In the yard Margaret waited, and Margaret waited, and then Margaret went inside at last. "
        "Only Hollis came late."
    )
    items = [
        _item("p1", True, "character_swap", text=passage),
        _item("p0", False, text=common),
    ]
    attack = evaluate.CorpusPriorAttack()
    attack.fit(items + [_item(f"q{n}", False, text=f"Then Hollis said it again, number {n}.") for n in range(30)])
    assert attack(items[0]).has_error is True


def test_pair_leak_localizes_the_edit_exactly():
    clean = "one two three\n\nfour five six\n\nseven eight nine"
    injected = clean.replace("five", "NINETY")
    items = [
        _item("p1", True, "r", paragraph=1, text=injected),
        _item("p1", False, text=clean),
    ]
    attack = evaluate.PairLeakAttack()
    attack.fit(items)
    assert attack(items[0]).paragraph_index == 1


def test_load_items_keeps_both_halves_when_filtering_by_rule(tmp_path=None):
    import json
    import tempfile

    rows = [
        _item("p1", True, "character_swap"),
        _item("p1", False),
        _item("p2", True, "trait_flip"),
        _item("p2", False),
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
        path = fh.name
    kept = evaluate.load_items(path, rule="trait_flip")
    assert len(kept) == 2
    assert {r["ground_truth"]["has_error"] for r in kept} == {True, False}


class _FakeUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text, input_tokens=1000, output_tokens=50):
        self.content = [_FakeBlock(text)]
        self.usage = _FakeUsage(input_tokens, output_tokens)


class _FakeMessages:
    def __init__(self, text):
        self.text = text
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse(self.text)


class _FakeClient:
    def __init__(self, text):
        self.messages = _FakeMessages(text)


def _llm(text, **kw):
    predictor = evaluate.LLMPredictor(**kw)
    predictor._client = _FakeClient(text)
    return predictor


def test_llm_predictor_reads_a_well_formed_verdict():
    predictor = _llm('{"has_error": true, "paragraph_index": 4, "reason": "eye colour changes"}')
    prediction = predictor(_item("p1", True, "r"))
    assert prediction.has_error is True
    assert prediction.paragraph_index == 4
    assert "eye colour" in prediction.note
    assert predictor.parse_failures == 0


def test_llm_predictor_maps_negative_paragraph_to_none():
    predictor = _llm('{"has_error": false, "paragraph_index": -1, "reason": "consistent"}')
    prediction = predictor(_item("p1", False))
    assert prediction.has_error is False
    assert prediction.paragraph_index is None


def test_llm_predictor_counts_a_parse_failure_instead_of_crashing():
    predictor = _llm("I think the passage is fine, actually.")
    prediction = predictor(_item("p1", True, "r"))
    assert prediction.has_error is False
    assert predictor.parse_failures == 1
    assert prediction.note.startswith("PARSE FAILURE")
    assert prediction.cost_usd > 0


def test_llm_predictor_prices_from_the_published_rates():
    predictor = _llm("{}", model="claude-opus-5")
    assert round(predictor.price(1_000_000, 0), 6) == 5.0
    assert round(predictor.price(0, 1_000_000), 6) == 25.0
    cheap = _llm("{}", model="claude-haiku-4-5")
    assert round(cheap.price(1_000_000, 1_000_000), 6) == 6.0


def test_llm_predictor_prices_an_unknown_model_as_zero_rather_than_guessing():
    predictor = _llm("{}", model="some-unreleased-model")
    assert predictor.price(1_000_000, 1_000_000) == 0.0


def test_llm_predictor_numbers_paragraphs_for_localization():
    predictor = _llm("{}")
    item = _item("p1", True, "r", text="first para\n\nsecond para\n\nthird para")
    prompt = predictor.build_prompt(item)
    assert "[0] first para" in prompt
    assert "[1] second para" in prompt
    assert "[2] third para" in prompt


def test_llm_predictor_sends_the_documented_request_shape():
    predictor = _llm('{"has_error": false, "paragraph_index": -1, "reason": "ok"}', effort="low")
    predictor(_item("p1", False))
    kwargs = predictor._client.messages.last_kwargs
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"]["effort"] == "low"
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert "budget_tokens" not in kwargs
    assert kwargs["output_config"]["format"]["schema"]["required"] == [
        "has_error",
        "paragraph_index",
        "reason",
    ]


def test_llm_predictor_is_registered_as_needing_a_key():
    assert evaluate.PREDICTORS["llm"].needs_key is True
    assert all(
        not predict.needs_key
        for name, predict in evaluate.PREDICTORS.items()
        if name != "llm"
    )


if __name__ == "__main__":
    import _selftest

    raise SystemExit(1 if _selftest.run(vars().copy()) else 0)
