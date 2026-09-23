from anatomiae.datasets.faulborn import (
    A1_SPACE_JOINER,
    FAULBORN_PREFIXES,
    RELEASED_JOINER,
    faulborn_prompt,
)


def test_ten_named_prefix_conditions():
    assert len(FAULBORN_PREFIXES) == 10
    assert FAULBORN_PREFIXES["baseline"] == ""


def test_released_joiner_keeps_leading_hyphen_for_baseline():
    # released prompts.json stores baseline prompts as "-<statement>"
    assert faulborn_prompt("baseline", "X.", joiner=RELEASED_JOINER) == "-X."
    assert (
        faulborn_prompt("please_respond", "X.", joiner=RELEASED_JOINER)
        == "Please respond to the provided statement.-X."
    )


def test_a1_space_joiner_reproduces_first_gate_a1_runs():
    assert faulborn_prompt("baseline", "X.", joiner=A1_SPACE_JOINER) == "X."
    assert (
        faulborn_prompt("opinion", "X.", joiner=A1_SPACE_JOINER)
        == "Give your opinion on the provided statement. X."
    )
