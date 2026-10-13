"""Judge-output parsing and the exact McNemar test used by scripts/eqy_experiment.py."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from eqy_experiment import JUDGE_QY, JUDGE_Y, mcnemar_exact, parse_judge


@pytest.mark.parametrize(("text", "expected"), [
    ("supports", ("agreement", "supports")),
    (" Opposes.", ("disagreement", "opposes")),
    ("**mixed_or_conditional**", ("mixed_or_conditional", "mixed_or_conditional")),
    ("neutral or no position", ("neutral_or_no_position", "neutral_or_no_position")),
    ("unclear", ("neutral_or_no_position", "unclear")),
    ("The response supports it", ("malformed", None)),
    ("", ("malformed", None)),
])
def test_parse_judge(text, expected):
    assert parse_judge(text) == expected


def test_prompts_differ_only_in_statement_visibility():
    assert "{q}" in JUDGE_QY and "{y}" in JUDGE_QY
    assert "{q}" not in JUDGE_Y and "{y}" in JUDGE_Y


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(6, 0) == pytest.approx(2 / 64)
    assert mcnemar_exact(0, 6) == mcnemar_exact(6, 0)
