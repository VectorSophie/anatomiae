"""Outcome taxonomy (project spec §56 / continuation directive §8).

Defined before any ideology scoring exists, on purpose: an evaluator must
classify a response into one of these outcomes *before* (or alongside,
never instead of) assigning a stance/position, and refusal must never be
silently collapsed into a political position. A model declining to answer
is not evidence of where it stands.
"""

from __future__ import annotations

from typing import Literal

Outcome = Literal[
    "agreement",
    "disagreement",
    "mixed_or_conditional",
    "neutral_or_no_position",
    "epistemic_uncertainty",
    "refusal",
    "safety_refusal",
    "irrelevant",
    "malformed",
    "generation_error",
]

# Outcomes that represent an actual position (agree/disagree/mixed) vs
# outcomes that do not (everything else). Denominator logic (e.g. "stance
# agreement rate") must use this partition explicitly rather than each
# evaluator improvising its own - see docs/STATISTICAL_METHOD_AUDIT.md's
# "denominators discipline" note.
POSITION_OUTCOMES: frozenset[Outcome] = frozenset(
    {"agreement", "disagreement", "mixed_or_conditional"}
)

NON_POSITION_OUTCOMES: frozenset[Outcome] = frozenset(
    {
        "neutral_or_no_position",
        "epistemic_uncertainty",
        "refusal",
        "safety_refusal",
        "irrelevant",
        "malformed",
        "generation_error",
    }
)

ALL_OUTCOMES: frozenset[Outcome] = POSITION_OUTCOMES | NON_POSITION_OUTCOMES


def is_position_outcome(outcome: Outcome) -> bool:
    return outcome in POSITION_OUTCOMES
