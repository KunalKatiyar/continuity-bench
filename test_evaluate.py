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


class _Stub(evaluate.VerdictPredictor):
    """A provider whose completion is supplied by the test."""

    provider = "stub"

    def __init__(self, text, input_tokens=1000, output_tokens=50, **kw):
        super().__init__(model=kw.pop("model", "claude-opus-5"), **kw)
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def complete(self, prompt):
        self.last_prompt = prompt
        return self.text, self.input_tokens, self.output_tokens


def test_verdict_predictor_reads_a_well_formed_verdict():
    predictor = _Stub('{"has_error": true, "paragraph_index": 4, "reason": "eye colour changes"}')
    prediction = predictor(_item("p1", True, "r"))
    assert prediction.has_error is True
    assert prediction.paragraph_index == 4
    assert "eye colour" in prediction.note
    assert predictor.parse_failures == 0


def test_verdict_predictor_maps_negative_paragraph_to_none():
    predictor = _Stub('{"has_error": false, "paragraph_index": -1, "reason": "consistent"}')
    prediction = predictor(_item("p1", False))
    assert prediction.has_error is False
    assert prediction.paragraph_index is None


def test_verdict_predictor_counts_a_parse_failure_instead_of_crashing():
    predictor = _Stub("I think the passage is fine, actually.")
    prediction = predictor(_item("p1", True, "r"))
    assert prediction.has_error is False
    assert predictor.parse_failures == 1
    assert predictor.parse_failure_notes == ["I think the passage is fine, actually."]
    assert prediction.note.startswith("PARSE FAILURE")
    assert prediction.cost_usd > 0


def test_pricing_comes_from_the_published_rates():
    predictor = _Stub("{}", model="claude-opus-5")
    assert round(predictor.price(1_000_000, 0), 6) == 5.0
    assert round(predictor.price(0, 1_000_000), 6) == 25.0
    cheap = _Stub("{}", model="claude-haiku-4-5")
    assert round(cheap.price(1_000_000, 1_000_000), 6) == 6.0


def test_an_unknown_model_reports_zero_cost_rather_than_a_guess():
    predictor = _Stub("{}", model="some-unreleased-model")
    assert predictor.has_rates() is False
    assert predictor.price(1_000_000, 1_000_000) == 0.0


def test_supplied_rates_override_the_table_and_cover_unknown_models():
    predictor = _Stub("{}", model="some-unreleased-model", rates=(3.0, 9.0))
    assert predictor.has_rates() is True
    assert round(predictor.price(1_000_000, 1_000_000), 6) == 12.0


def test_paragraphs_are_numbered_for_localization():
    predictor = _Stub("{}")
    item = _item("p1", True, "r", text="first para\n\nsecond para\n\nthird para")
    prompt = predictor.build_prompt(item)
    assert "[0] first para" in prompt
    assert "[1] second para" in prompt
    assert "[2] third para" in prompt


def test_both_providers_share_one_prompt_and_one_schema():
    """A cross-provider comparison is meaningless if each provider sees a different prompt."""
    claude = evaluate.AnthropicPredictor("claude-opus-5")
    openai = evaluate.OpenAIPredictor("gpt-4.1")
    item = _item("p1", True, "r", text="a para here\n\nb para here")
    assert claude.build_prompt(item) == openai.build_prompt(item)
    assert evaluate.VERDICT_JSON_SCHEMA["required"] == [
        "has_error",
        "paragraph_index",
        "reason",
    ]
    assert evaluate.VERDICT_JSON_SCHEMA["additionalProperties"] is False


def test_provider_names_are_distinct_so_results_do_not_overwrite():
    claude = evaluate.AnthropicPredictor("claude-opus-5", effort="high")
    openai = evaluate.OpenAIPredictor("gpt-4.1", effort="high")
    assert claude.predictor_name == "claude_claude-opus-5_high"
    assert openai.predictor_name == "openai_gpt-4.1_high"
    assert claude.predictor_name != openai.predictor_name


def test_anthropic_request_matches_the_documented_shape():
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            block = type("B", (), {"type": "text", "text": '{"has_error": false, "paragraph_index": -1, "reason": "ok"}'})()
            usage = type("U", (), {"input_tokens": 10, "output_tokens": 2})()
            return type("R", (), {"content": [block], "usage": usage})()

    predictor = evaluate.AnthropicPredictor("claude-opus-5", effort="low")
    predictor._client = type("C", (), {"messages": FakeMessages()})()
    predictor(_item("p1", False))
    assert captured["model"] == "claude-opus-5"
    assert captured["thinking"] == {"type": "adaptive"}
    assert captured["output_config"]["effort"] == "low"
    assert captured["output_config"]["format"]["type"] == "json_schema"
    assert "budget_tokens" not in captured


def test_openai_request_uses_a_strict_json_schema_response_format():
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = type("M", (), {"content": '{"has_error": true, "paragraph_index": 1, "reason": "ok"}'})()
            choice = type("C", (), {"message": message})()
            usage = type("U", (), {"prompt_tokens": 20, "completion_tokens": 4})()
            return type("R", (), {"choices": [choice], "usage": usage})()

    predictor = evaluate.OpenAIPredictor("gpt-4.1")
    predictor._client = type(
        "C", (), {"chat": type("Ch", (), {"completions": FakeCompletions()})()}
    )()
    prediction = predictor(_item("p1", True, "r"))
    assert captured["model"] == "gpt-4.1"
    fmt = captured["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == evaluate.VERDICT_JSON_SCHEMA
    assert [m["role"] for m in captured["messages"]] == ["system", "user"]
    assert prediction.has_error is True
    assert prediction.input_tokens == 20 and prediction.output_tokens == 4
    assert round(prediction.cost_usd, 8) == round(20 / 1e6 * 2.0 + 4 / 1e6 * 8.0, 8)


def test_every_paid_predictor_is_registered_and_flagged():
    for name in ("llm", "openai", "jev", "jev_hybrid"):
        assert evaluate.PREDICTORS[name].needs_key is True, name
        assert evaluate.PREDICTORS[name].kind == "model", name
    assert evaluate.PREDICTORS["local"].needs_key is False
    assert all(
        not predict.needs_key
        for name, predict in evaluate.PREDICTORS.items()
        if predict.kind in evaluate.HEURISTIC_KINDS
    )


def test_jev_is_priced_input_only_because_it_generates_nothing():
    jev = evaluate.PREDICTORS["jev"]
    assert jev.rates == (0.042, 0.0)
    assert jev.price(1_000_000, 1_000_000) == 0.042
    assert round(jev.price(607_000, 0), 4) == 0.0255


def test_a_missing_key_file_leaves_the_sdk_to_use_the_environment():
    assert evaluate.read_key_file("~/.definitely-not-a-key-file-9f3a") is None


class _Boom(evaluate.VerdictPredictor):
    """A provider whose API call always raises the given exception."""

    provider = "boom"

    def __init__(self, exc, **kw):
        super().__init__(model=kw.pop("model", "claude-opus-5"), **kw)
        self.exc = exc

    def complete(self, prompt):
        raise self.exc


class _FakeAPIError(Exception):
    def __init__(self, message, body=None, status_code=None):
        super().__init__(message)
        self.body = body
        self.status_code = status_code


def test_a_quota_failure_is_fatal_so_a_long_run_stops_at_once():
    exc = _FakeAPIError(
        "Error code: 429 - you have no credits remaining",
        body={"error": {"code": "insufficient_quota", "type": "insufficient_quota"}},
    )
    assert "insufficient_quota" in evaluate.classify_api_error(exc)
    predictor = _Boom(exc)
    try:
        predictor(_item("p1", True, "r"))
    except evaluate.RunAborted as abort:
        assert "insufficient_quota" in str(abort)
    else:
        raise AssertionError("a quota failure should abort the run")


def test_an_auth_failure_is_fatal():
    assert evaluate.classify_api_error(_FakeAPIError("nope", status_code=401))
    assert evaluate.classify_api_error(_FakeAPIError("nope", status_code=403))


def test_a_transient_failure_skips_one_item_instead_of_losing_the_run():
    predictor = _Boom(_FakeAPIError("Connection reset by peer", status_code=500))
    prediction = predictor(_item("p1", True, "r"))
    assert prediction.has_error is False
    assert predictor.api_failures == 1
    assert prediction.note.startswith("API FAILURE")


def test_a_transient_failure_is_not_classified_fatal():
    assert evaluate.classify_api_error(_FakeAPIError("overloaded", status_code=529)) is None
    assert evaluate.classify_api_error(_FakeAPIError("timed out")) is None


def test_skipped_items_still_count_against_recall():
    """A skipped item must not be silently dropped, or a broken run looks clean."""
    predictor = _Boom(_FakeAPIError("timed out"))
    items = [_item("p1", True, "r"), _item("p1", False)]
    scores = evaluate.score(items, predictor)
    assert scores.recall == 0.0
    assert scores.false_negative == 1
    assert predictor.api_failures == 2


def test_local_predictor_is_free_and_needs_no_key():
    predictor = evaluate.LocalPredictor("llama3.1:8b")
    assert predictor.needs_key is False
    assert predictor.has_rates() is True
    assert predictor.price(1_000_000, 1_000_000) == 0.0
    assert predictor.predictor_name == "local_llama3.1:8b_high"


def test_local_predictor_reuses_the_shared_prompt_and_schema():
    """A local baseline is only comparable if it is asked the same question."""
    local = evaluate.LocalPredictor("llama3.1:8b")
    hosted = evaluate.OpenAIPredictor("gpt-4.1")
    item = _item("p1", True, "r", text="a para here\n\nb para here")
    assert local.build_prompt(item) == hosted.build_prompt(item)
    assert local.build_prompt(item) == evaluate.AnthropicPredictor("claude-opus-5").build_prompt(item)


def test_local_predictor_points_at_ollama_by_default_and_accepts_an_override():
    assert evaluate.LocalPredictor("llama3.1:8b").base_url == "http://localhost:11434/v1"
    custom = evaluate.LocalPredictor("llama3.1:8b", base_url="http://gpu-box:8000/v1")
    assert custom.base_url == "http://gpu-box:8000/v1"


def test_a_free_local_run_does_not_create_a_cost_axis():
    """A local run costs no API money, so the leaderboard must stay on bars."""
    import build_site

    rows = [
        {"predictor": "local_llama3.1:8b_high", "kind": "model", "cost_usd": 0.0,
         "youden_j": 0.2, "f1": 0.5, "items": 40, "recall": 0.5,
         "false_positive_rate": 0.3, "localization": 0.1, "precision": 0.5,
         "novels": 5, "corpus": "c.jsonl"},
    ]
    assert build_site.has_paid_run(rows) is False


def test_the_attack_suite_never_includes_a_model():
    """`attack` must stay instant and free.

    Filtering on needs_key was not enough: a local model needs no key, so registering
    one put a 58-minute llama run inside the "zero-cost heuristic" suite. The filter is
    on kind.
    """
    for name in ("llm", "openai", "local"):
        assert evaluate.PREDICTORS[name].kind == "model"
        assert evaluate.PREDICTORS[name].kind not in evaluate.HEURISTIC_KINDS
    heuristics = [n for n, p in evaluate.PREDICTORS.items() if p.kind in evaluate.HEURISTIC_KINDS]
    assert "local" not in heuristics
    assert {"always_clean", "corpus_prior", "pair_leak"} <= set(heuristics)


def test_every_heuristic_is_genuinely_free():
    for name, predict in evaluate.PREDICTORS.items():
        if predict.kind in evaluate.HEURISTIC_KINDS:
            assert predict.needs_key is False, name


def test_a_tight_interval_around_zero_is_a_measured_null_not_an_unknown():
    """A narrow CI containing zero and a wide one mean opposite things."""
    assert evaluate.j_verdict(-0.039, 0.048) == "no_effect"
    assert evaluate.j_verdict(-0.438, 0.271) == "inconclusive"
    assert evaluate.j_verdict(0.30, 0.55) == "effect"
    assert evaluate.j_verdict(-0.55, -0.30) == "effect"


def test_the_verdict_travels_into_the_results_record():
    items = []
    for n in range(40):
        items += [_item(f"p{n}", True, "r"), _item(f"p{n}", False)]
    summary = evaluate.score(items, _fixed(True, 3)).summary()
    assert summary["youden_j"] == 0.0
    assert summary["youden_j_verdict"] in ("no_effect", "inconclusive")
    assert summary["youden_j_significant"] is False


if __name__ == "__main__":
    import _selftest

    raise SystemExit(1 if _selftest.run(vars().copy()) else 0)
