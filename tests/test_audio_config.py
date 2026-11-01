from __future__ import annotations

import pytest
from pydantic import ValidationError

from atlas.config import AudioCfg, Config, load_config


def test_defaults_match_the_plan():
    a = load_config().audio
    assert (a.voice, a.model, a.language, a.bitrate, a.sample_rate, a.gap_ms) == ("alba", "kyutai/pocket-tts", "english", "48k", 24000, 600)
    assert (a.top_overall, a.per_category, a.categories, a.max_papers, a.retention_days) == (20, 3, 12, 40, 8)
    assert (a.align, a.whisper_model, a.edge_ms, a.sentence_gap_ms) == ("whisper", "tiny.en", 120, 250)
    assert str(a.enabled) == "auto" and a.max_seconds > 0 and a.max_site_mb > 0
    assert (a.summary.provider, a.summary.model, a.summary.words_min, a.summary.words_max) == (None, None, 55, 85)
    assert 0 < a.summary.max_cost_usd_per_run <= 0.25


def test_config_yaml_and_code_defaults_agree():
    assert load_config().audio == AudioCfg()


def test_unknown_audio_keys_are_rejected_like_everywhere_else():
    with pytest.raises(ValidationError):
        Config.model_validate({"audio": {"voicee": "alba"}})
    with pytest.raises(ValidationError):
        Config.model_validate({"audio": {"summary": {"wordz": 1}}})
    with pytest.raises(ValidationError):
        Config.model_validate({"audio": {"align": "magic"}})


def test_the_config_round_trips_through_json_for_the_child_process():
    cfg = load_config()
    import json
    import yaml
    again = Config.model_validate(yaml.safe_load(json.dumps(cfg.model_dump(mode="json", by_alias=True))))
    assert again == cfg and again.media.queue.schema_name == cfg.media.queue.schema_name


def test_the_audio_budget_fits_inside_the_site_budget():
    cfg = load_config()
    assert cfg.audio.max_site_mb + cfg.media.max_site_mb < cfg.output.max_site_mb
