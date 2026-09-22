from anatomiae.models.registry import load_registry

LOCKED_ANCHOR_FAMILIES = {"olmo2", "amber"}


def test_registry_loads_all_configs():
    reg = load_registry()
    assert len(reg) >= 11


def test_olmo2_lineage_is_a_connected_chain():
    reg = load_registry()
    base = reg["olmo2-13b-base"]
    sft = reg["olmo2-13b-sft"]
    dpo = reg["olmo2-13b-dpo"]
    rlvr = reg["olmo2-13b-instruct-rlvr2"]

    assert base.parent_checkpoint is None
    assert sft.parent_checkpoint == "olmo2-13b-base"
    assert dpo.parent_checkpoint == "olmo2-13b-sft"
    assert rlvr.parent_checkpoint == "olmo2-13b-dpo"


def test_qwen_deepseek_lineage_shares_a_common_parent():
    reg = load_registry()
    qwen_instruct = reg["qwen2.5-14b-instruct"]
    r1_distill = reg["deepseek-r1-distill-qwen-14b"]
    assert qwen_instruct.parent_checkpoint == r1_distill.parent_checkpoint == "qwen2.5-14b-base"


def test_locked_anchors_are_flagged():
    reg = load_registry()
    for name, entry in reg.items():
        family_is_locked_anchor = entry.family in LOCKED_ANCHOR_FAMILIES
        if family_is_locked_anchor and name != "olmo2-1b-instruct-smoke":
            assert entry.locked_anchor, f"{name} should be flagged as a locked anchor"


def test_no_duplicate_checkpoints():
    reg = load_registry()
    checkpoints = [(e.checkpoint, e.revision) for e in reg.values()]
    assert len(checkpoints) == len(set(checkpoints))
