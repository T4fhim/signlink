"""Training configs (`ml/configs/*.yaml`) and the data-track check (ADR-0003, CLAUDE.md rule 4)."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from signlnk_ml.features.normalize import REPO_ROOT
from signlnk_ml.training.augment import AugmentConfig

CONFIGS_DIR = REPO_ROOT / "ml" / "configs"
REGISTRY_PATH = REPO_ROOT / "ml" / "datasets.yaml"
TRACKS = ("release", "research")
T = TypeVar("T")


@dataclass(frozen=True)
class WindowSettings:
    length: int = 64
    other_min_run: int = 8
    other_ratio: float = 2.0


@dataclass(frozen=True)
class InputSettings:
    use_z: bool = False
    velocity: bool = False


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    embedding_dim: int = 256
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainSettings:
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    seed: int = 1


@dataclass(frozen=True)
class TrainConfig:
    track: str
    datasets: list[str]
    pretrained: list[str]
    splits: str
    window: WindowSettings
    input: InputSettings
    model: ModelConfig
    augment: AugmentConfig
    train: TrainSettings


def _build(cls: type[T], raw: Mapping[str, Any], where: str) -> T:
    unknown = sorted(set(raw) - {f.name for f in fields(cls)})  # type: ignore[arg-type]
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}")
    values = {key: tuple(value) if key.endswith("_range") else value for key, value in raw.items()}
    return cls(**values)


SECTIONS: dict[str, type[Any]] = {
    "window": WindowSettings,
    "input": InputSettings,
    "model": ModelConfig,
    "augment": AugmentConfig,
    "train": TrainSettings,
}


def apply_overrides(raw: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Applies `section.key=value` overrides (values parsed as YAML) to a copy of a raw config."""
    out: dict[str, Any] = copy.deepcopy(dict(raw))
    for item in overrides:
        key, separator, text = item.partition("=")
        if not separator or not key:
            raise ValueError(f"override must look like key=value, got {item!r}")
        *parents, leaf = key.split(".")
        node = out
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(text)
    return out


def load_raw(path: Path, overrides: Sequence[str] = ()) -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return apply_overrides(raw, overrides)


def load_config(path: Path, overrides: Sequence[str] = ()) -> TrainConfig:
    raw = load_raw(path, overrides)
    unknown = sorted(set(raw) - set(SECTIONS) - {"track", "datasets", "pretrained", "splits"})
    if unknown:
        raise ValueError(f"{path.name}: unknown key(s) {unknown}")
    built = {
        name: _build(cls, raw.get(name, {}), f"{path.name}:{name}")
        for name, cls in SECTIONS.items()
    }
    return TrainConfig(
        track=raw["track"],
        datasets=list(raw["datasets"]),
        pretrained=list(raw.get("pretrained") or []),
        splits=raw["splits"],
        **built,
    )


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return registry


def check_config(raw: Mapping[str, Any], registry: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Problems with a config's declared data track; empty means clean (ADR-0003)."""
    track = raw.get("track")
    if track not in TRACKS:
        return [f"track must be one of {TRACKS}, got {track!r}"]
    problems: list[str] = []
    datasets = list(raw.get("datasets") or [])
    if not datasets:
        problems.append("datasets is empty: list every dataset the config trains or evaluates on")
    for source in [*datasets, *(raw.get("pretrained") or [])]:
        entry = registry.get(source)
        if entry is None:
            problems.append(f"{source} is not in ml/datasets.yaml: run /dataset-license first")
        elif track == "release" and entry["track"] != "release":
            problems.append(f"release config uses {entry['track']}-track source {source}")
    return problems
