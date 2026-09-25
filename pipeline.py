#!/usr/bin/env python3
"""The Jev editing harness: extract story state, verify cheaply, escalate rarely.

    chapter text  --[LLM, once per chapter]-->  story state (facts, traits, timeline)
    paragraph     --[Jev, once per paragraph]-> contradicts state? yes/no + confidence
    flagged only  --[LLM, once per flag]------> confirm, and write the explanation

The shape of the bet: verification is a closed question ("does this paragraph
contradict a fact already in state?"), which is what a System One model answers in one
parallel pass for $0.042 per million input tokens with no output cost. Generation is
only needed for the two ends - building the state, and explaining a real hit to an
author - so the expensive model runs on a small fraction of the text.

Escalation is driven by Jev's calibrated confidence, not just its answer: a "no" it is
unsure about is worth a second look, and an unsure "yes" is where false positives live.

ponytail: state is per-passage here rather than incremental across a whole novel,
because that is what the benchmark's items are. Incremental per-chapter extraction is
the thing to build when this runs on a real manuscript - the interfaces below do not
change, only who calls extract_state and how often.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class ContradictionVerdict(BaseModel):
    """Decide whether this paragraph contradicts an established fact about the story."""

    contradicts_state: Annotated[
        bool,
        Field(
            description=(
                "true when the paragraph states something that cannot be true given the "
                "established facts; false when it is merely new, vague, or consistent"
            )
        ),
    ]
    """Does this paragraph contradict a fact listed in the story state?"""


class EscalationVerdict(BaseModel):
    """An LLM's confirmation of a flagged paragraph, and the note an author reads."""

    is_real_contradiction: bool
    paragraph_index: int
    explanation: str


@dataclass
class StoryState:
    """The facts a later paragraph can contradict.

    Deliberately small and flat: it is prompt context for every paragraph check, so its
    size is multiplied by the number of paragraphs and is the main cost lever here.
    """

    characters: dict = field(default_factory=dict)
    timeline: list = field(default_factory=list)
    established_facts: list = field(default_factory=list)

    def as_prompt(self):
        """Render the state as the block a verifier is asked to check against."""
        lines = []
        for name, traits in sorted(self.characters.items()):
            described = ", ".join(f"{k}: {v}" for k, v in sorted(traits.items()))
            lines.append(f"- {name}: {described}")
        for entry in self.timeline:
            lines.append(f"- timeline: {entry}")
        for fact in self.established_facts:
            lines.append(f"- {fact}")
        return "\n".join(lines) if lines else "(no facts established yet)"

    def is_empty(self):
        return not (self.characters or self.timeline or self.established_facts)

    @classmethod
    def from_json(cls, payload):
        """Build state from an extraction pass, tolerating a partial or odd shape."""
        if not isinstance(payload, dict):
            return cls()
        characters = payload.get("characters")
        timeline = payload.get("timeline")
        facts = payload.get("established_facts")
        return cls(
            characters=characters if isinstance(characters, dict) else {},
            timeline=[str(t) for t in timeline] if isinstance(timeline, list) else [],
            established_facts=[str(f) for f in facts] if isinstance(facts, list) else [],
        )


EXTRACTION_PROMPT = """Read this passage of fiction and list the facts a later
paragraph could contradict. Record only what the text states or clearly implies - never
infer a trait that is not there, and never invent one to fill a field.

Return JSON with:
  characters: an object mapping each named character to an object of stated traits
              (eye_colour, hair_colour, age, relationships, anything stated)
  timeline:   a list of stated time markers in the order they occur
  established_facts: a list of short factual statements that later text could contradict

Passage:

{text}"""

ESCALATION_PROMPT = """A cheap classifier flagged one paragraph of a passage as
contradicting an established fact. Confirm or reject that, and if it is real, write the
one-sentence note an author would read.

Established facts:
{state}

Flagged paragraph (index {index}):
{paragraph}

Full passage for context:
{text}

Answer with is_real_contradiction, paragraph_index, and explanation."""


@dataclass
class Finding:
    """One flagged paragraph, with how it was reached."""

    paragraph_index: int
    jev_says_contradiction: bool
    jev_confidence: float
    escalated: bool = False
    confirmed: bool | None = None
    explanation: str = ""


@dataclass
class PipelineResult:
    findings: list = field(default_factory=list)
    paragraphs_checked: int = 0
    escalations: int = 0
    jev_input_tokens: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0

    @property
    def confirmed(self):
        return [f for f in self.findings if f.confirmed]

    @property
    def escalation_rate(self):
        return self.escalations / self.paragraphs_checked if self.paragraphs_checked else 0.0


class JevVerifier:
    """Wraps Jev: one typed question per paragraph, answered with a confidence.

    Jev returns a calibrated probability rather than a sampled token, which is what
    makes a confidence-banded escalation policy possible at all. Credentials come from
    TYPESAFE_API_KEY, or ~/.typesafe-key.
    """

    def __init__(self, model="typesafe:jev-latest"):
        self.model = model
        self._agent = None
        self.input_tokens = 0

    def agent(self):
        if self._agent is None:
            from pydantic_ai import Agent

            self._ensure_key()
            self._agent = Agent(self.model, output_type=ContradictionVerdict)
        return self._agent

    @staticmethod
    def _ensure_key():
        """Let a key file stand in for the env var, so no key is pasted into a session."""
        if os.environ.get("TYPESAFE_API_KEY"):
            return
        from pathlib import Path

        key_file = Path.home() / ".typesafe-key"
        if key_file.exists():
            key = key_file.read_text(encoding="utf-8").strip()
            if key:
                os.environ["TYPESAFE_API_KEY"] = key

    def check(self, state, paragraph):
        """Return (contradicts, confidence) for one paragraph against the state."""
        prompt = (
            f"Established facts:\n{state.as_prompt()}\n\n"
            f"Paragraph to check:\n{paragraph}"
        )
        result = self.agent().run_sync(prompt)
        usage = getattr(result, "usage", None)
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        confidence = 0.0
        details = getattr(getattr(result, "response", None), "provider_details", None)
        if isinstance(details, dict):
            scores = details.get("confidence")
            if isinstance(scores, dict):
                confidence = float(scores.get("contradicts_state", 0.0))
            elif isinstance(scores, (int, float)):
                confidence = float(scores)
        return bool(result.output.contradicts_state), confidence


def run_pipeline(
    text,
    verifier,
    extract,
    escalate,
    low_confidence=0.75,
    state=None,
):
    """Run extract -> verify -> escalate over one passage and return a PipelineResult.

    extract(text) -> StoryState and escalate(state, index, paragraph, text) ->
    EscalationVerdict are injected, so the harness can be exercised without any network
    and so the LLM behind them is a choice rather than a hard dependency.

    A paragraph is escalated when Jev says contradiction, or when Jev says no but is not
    confident. Skipping the unsure "no" would quietly drop exactly the cases a cheap
    gate is worst at.
    """
    paragraphs = text.split("\n\n")
    result = PipelineResult()
    state = state if state is not None else extract(text)
    for index, paragraph in enumerate(paragraphs):
        contradicts, confidence = verifier.check(state, paragraph)
        result.paragraphs_checked += 1
        unsure = confidence < low_confidence
        if not contradicts and not unsure:
            continue
        finding = Finding(index, contradicts, confidence)
        verdict = escalate(state, index, paragraph, text)
        finding.escalated = True
        result.escalations += 1
        if verdict is not None:
            finding.confirmed = bool(verdict.is_real_contradiction)
            finding.explanation = verdict.explanation
        result.findings.append(finding)
    result.jev_input_tokens = verifier.input_tokens
    return result


def parse_state(raw):
    """Parse an extraction response into StoryState, tolerating fenced JSON."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return StoryState.from_json(json.loads(text))
    except json.JSONDecodeError:
        return StoryState()
