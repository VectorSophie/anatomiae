import pytest
from pydantic import ValidationError

from anatomiae.datasets.schema import DatasetItem


def _item(**overrides):
    defaults = {
        "item_id": "test-1",
        "source_dataset": "test",
        "source_version": "v1",
        "issue": "taxation",
        "dimension": "economic",
        "language": "en",
        "translation_origin": "native",
        "independently_authored": True,
        "prompt_original": "Should taxes be raised?",
    }
    defaults.update(overrides)
    return DatasetItem(**defaults)


def test_native_item_requires_no_translation_fields():
    item = _item()
    assert item.source_language is None
    assert item.translation_origin == "native"
    assert item.independently_authored is True


def test_translation_origin_is_mandatory():
    with pytest.raises(ValidationError):
        DatasetItem(
            item_id="x",
            source_dataset="test",
            source_version="v1",
            issue="taxation",
            dimension="economic",
            language="en",
            independently_authored=True,
            prompt_original="...",
        )


def test_lim_rottger_style_translated_pair_is_representable():
    """Regression case: the Lim & Roettger bilingual dataset pairs languages
    via human translation from a single-language source, not independent
    per-language authorship - the schema must be able to say so explicitly,
    not silently imply two cultures independently produced parallel items."""
    en_item = _item(
        item_id="lr-42-zh-source",
        language="zh",
        source_language="zh",
        translation_origin="native",
        independently_authored=True,
        prompt_original="中文提示",
        parallel_item_id="lr-42-en-translated",
    )
    translated_item = _item(
        item_id="lr-42-en-translated",
        language="en",
        source_language="zh",
        translation_origin="human_translation",
        translator_type="paper_author",
        independently_authored=False,
        prompt_original="English prompt translated from Chinese source media",
        parallel_item_id="lr-42-zh-source",
        translation_notes="Lim & Roettger (2026): issue sourced from Chinese media, manually translated to English.",
    )
    assert en_item.independently_authored is True
    assert translated_item.independently_authored is False
    assert translated_item.translation_origin == "human_translation"
    assert translated_item.parallel_item_id == en_item.item_id
