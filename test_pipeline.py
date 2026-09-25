"""Self-checks for pipeline.py, the Jev editing harness. No network."""

import pipeline


class _FakeVerifier:
    """Answers each paragraph from a scripted list of (contradicts, confidence)."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.input_tokens = 0
        self.seen = []

    def check(self, state, paragraph):
        self.seen.append(paragraph)
        self.input_tokens += 100
        return self.answers.pop(0) if self.answers else (False, 1.0)


def _passage(n=4):
    return "\n\n".join(f"paragraph {i} text" for i in range(n))


def _extract(_text):
    return pipeline.StoryState(characters={"Dora": {"eye_colour": "blue"}})


def _escalate_confirming(state, index, paragraph, text):
    return pipeline.EscalationVerdict(
        is_real_contradiction=True, paragraph_index=index, explanation="eye colour changes"
    )


def _escalate_rejecting(state, index, paragraph, text):
    return pipeline.EscalationVerdict(
        is_real_contradiction=False, paragraph_index=index, explanation="not a contradiction"
    )


def test_state_renders_facts_for_the_verifier():
    state = pipeline.StoryState(
        characters={"Dora": {"eye_colour": "blue"}},
        timeline=["morning, day 1"],
        established_facts=["the letter was burned"],
    )
    rendered = state.as_prompt()
    assert "Dora: eye_colour: blue" in rendered
    assert "timeline: morning, day 1" in rendered
    assert "the letter was burned" in rendered
    assert state.is_empty() is False


def test_empty_state_says_so_rather_than_rendering_nothing():
    assert pipeline.StoryState().as_prompt() == "(no facts established yet)"
    assert pipeline.StoryState().is_empty() is True


def test_extraction_output_is_parsed_and_a_bad_shape_does_not_crash():
    assert pipeline.parse_state('{"characters": {"A": {"age": 30}}}').characters == {"A": {"age": 30}}
    assert pipeline.parse_state('```json\n{"timeline": ["day 1"]}\n```').timeline == ["day 1"]
    assert pipeline.parse_state("not json at all").is_empty()
    assert pipeline.parse_state('["a list, not an object"]').is_empty()
    assert pipeline.parse_state('{"characters": "wrong type"}').characters == {}


def test_only_flagged_paragraphs_reach_the_expensive_model():
    """The whole economic claim: the LLM must not see every paragraph."""
    verifier = _FakeVerifier([(False, 0.99), (True, 0.95), (False, 0.99), (False, 0.98)])
    escalated = []

    def escalate(state, index, paragraph, text):
        escalated.append(index)
        return pipeline.EscalationVerdict(
            is_real_contradiction=True, paragraph_index=index, explanation="why"
        )

    result = pipeline.run_pipeline(_passage(), verifier, _extract, escalate)
    assert result.paragraphs_checked == 4
    assert escalated == [1]
    assert result.escalations == 1
    assert result.escalation_rate == 0.25


def test_an_unsure_no_is_escalated_too():
    """A cheap gate is worst exactly where it is unsure, so an unsure no is not dropped."""
    verifier = _FakeVerifier([(False, 0.40), (False, 0.99), (False, 0.99), (False, 0.99)])
    result = pipeline.run_pipeline(_passage(), verifier, _extract, _escalate_rejecting)
    assert result.escalations == 1
    assert result.findings[0].paragraph_index == 0
    assert result.findings[0].jev_says_contradiction is False


def test_the_confidence_threshold_is_adjustable():
    answers = [(False, 0.40), (False, 0.99), (False, 0.99), (False, 0.99)]
    strict = pipeline.run_pipeline(
        _passage(), _FakeVerifier(answers), _extract, _escalate_rejecting, low_confidence=0.0
    )
    assert strict.escalations == 0


def test_escalation_can_reject_a_jev_false_positive():
    verifier = _FakeVerifier([(True, 0.95), (False, 0.99), (False, 0.99), (False, 0.99)])
    result = pipeline.run_pipeline(_passage(), verifier, _extract, _escalate_rejecting)
    assert result.escalations == 1
    assert result.findings[0].confirmed is False
    assert result.confirmed == []


def test_a_confirmed_finding_carries_the_authors_explanation():
    verifier = _FakeVerifier([(True, 0.95), (False, 0.99), (False, 0.99), (False, 0.99)])
    result = pipeline.run_pipeline(_passage(), verifier, _extract, _escalate_confirming)
    assert len(result.confirmed) == 1
    assert result.confirmed[0].explanation == "eye colour changes"
    assert result.confirmed[0].jev_confidence == 0.95


def test_every_paragraph_is_verified_against_the_state():
    verifier = _FakeVerifier([(False, 0.99)] * 4)
    pipeline.run_pipeline(_passage(4), verifier, _extract, _escalate_rejecting)
    assert len(verifier.seen) == 4
    assert verifier.seen[2] == "paragraph 2 text"


def test_extraction_runs_once_per_passage_not_once_per_paragraph():
    calls = []

    def counting_extract(text):
        calls.append(text)
        return pipeline.StoryState()

    pipeline.run_pipeline(_passage(6), _FakeVerifier([(False, 0.99)] * 6), counting_extract, _escalate_rejecting)
    assert len(calls) == 1


def test_a_supplied_state_skips_extraction_entirely():
    """On a real manuscript the state is built per chapter and reused across paragraphs."""

    def exploding_extract(text):
        raise AssertionError("extraction should not run when state is supplied")

    result = pipeline.run_pipeline(
        _passage(), _FakeVerifier([(False, 0.99)] * 4), exploding_extract,
        _escalate_rejecting, state=pipeline.StoryState(established_facts=["x"]),
    )
    assert result.paragraphs_checked == 4


def test_jev_token_use_is_recorded_for_costing():
    verifier = _FakeVerifier([(False, 0.99)] * 4)
    result = pipeline.run_pipeline(_passage(), verifier, _extract, _escalate_rejecting)
    assert result.jev_input_tokens == 400


if __name__ == "__main__":
    import _selftest

    raise SystemExit(1 if _selftest.run(vars().copy()) else 0)
