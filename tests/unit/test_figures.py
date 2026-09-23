import pytest

pytest.importorskip("matplotlib")  # `analysis` extra; CI runs `dev` only
pd = pytest.importorskip("pandas")

from anatomiae.evaluators.taxonomy import ALL_OUTCOMES
from anatomiae.plots.figures import OUTCOME_COLORS, outcome_stack


def test_every_taxonomy_outcome_has_a_fixed_color():
    # regression: `malformed` was once missing and would have been silently
    # dropped from 100%-stacked bars
    assert set(ALL_OUTCOMES) <= set(OUTCOME_COLORS)


def test_outcome_stack_refuses_unmapped_outcomes():
    long = pd.DataFrame({"evaluator": ["a", "a"], "outcome": ["agreement", "not_an_outcome"]})
    with pytest.raises(ValueError, match="without a fixed color"):
        outcome_stack(long, group_col="evaluator", title="t", subtitle="s")
