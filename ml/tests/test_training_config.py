"""Training configs and the data-track check (ADR-0003, Phase 1 step 3)."""

from pathlib import Path

import pytest
import yaml
from signlnk_ml.features.normalize import REPO_ROOT
from signlnk_ml.training.config import (
    CONFIGS_DIR,
    REGISTRY_PATH,
    check_config,
    load_config,
    load_registry,
)

REGISTRY = {
    "kaggle-islr": {"track": "release"},
    "asl-citizen": {"track": "research"},
}


def config(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {"track": "release", "datasets": ["kaggle-islr"], "pretrained": []}
    return base | overrides


def test_a_clean_release_config_passes() -> None:
    assert check_config(config(), REGISTRY) == []


def test_missing_or_invalid_track_is_reported() -> None:
    assert check_config({"datasets": ["kaggle-islr"]}, REGISTRY)
    assert check_config(config(track="public"), REGISTRY)


def test_release_config_with_research_data_is_reported() -> None:
    problems = check_config(config(datasets=["kaggle-islr", "asl-citizen"]), REGISTRY)
    assert any("asl-citizen" in p and "research" in p for p in problems)


def test_research_config_may_use_both_tracks() -> None:
    both = config(track="research", datasets=["kaggle-islr", "asl-citizen"])
    assert check_config(both, REGISTRY) == []


def test_unknown_dataset_or_pretrained_source_is_reported() -> None:
    assert any("mystery" in p for p in check_config(config(datasets=["mystery"]), REGISTRY))
    assert any("imagenet" in p for p in check_config(config(pretrained=["imagenet"]), REGISTRY))


def test_release_config_with_a_research_pretrained_source_is_reported() -> None:
    assert check_config(config(pretrained=["asl-citizen"]), REGISTRY)


def test_empty_dataset_list_is_reported() -> None:
    assert check_config(config(datasets=[]), REGISTRY)


def test_committed_registry_lists_every_dataset_in_plan_5_1() -> None:
    registry = load_registry(REGISTRY_PATH)
    assert registry["kaggle-islr"]["track"] == "release"
    assert registry["studio-recordings"]["track"] == "release"
    for research in ("asl-citizen", "sem-lex", "wlasl", "how2sign"):
        assert registry[research]["track"] == "research"


def test_every_committed_training_config_passes_the_track_check() -> None:
    registry = load_registry(REGISTRY_PATH)
    paths = sorted(CONFIGS_DIR.glob("*.yaml"))
    assert paths, "no training config committed"
    for path in paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert check_config(raw, registry) == [], path.name


def test_baseline_config_loads_with_expected_choices() -> None:
    cfg = load_config(CONFIGS_DIR / "baseline.yaml")
    assert cfg.track == "release" and cfg.datasets == ["kaggle-islr"]
    assert cfg.window.length == 64 and cfg.model.embedding_dim == 256
    assert (REPO_ROOT / cfg.splits).exists()
    assert cfg.input.use_z is False  # hand z span differs from Kaggle (PROJECT_CONTEXT [VERIFY])


def test_unknown_config_keys_are_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load((CONFIGS_DIR / "baseline.yaml").read_text(encoding="utf-8"))
    raw["train"]["epochz"] = 3
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="epochz"):
        load_config(bad)
