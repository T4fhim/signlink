"""Train/serve distribution parity for landmark sequences.

Compares per-landmark statistics of normalized landmarks from two sources (for example Kaggle ISLR
mapped from Holistic vs a browser recording from MediaPipe Tasks). A wrong index, a left/right swap
or a coordinate-convention difference shows up as a large shift in some landmarks' mean.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from signlnk_ml.data.geometry import group_slice, pose_position
from signlnk_ml.features.normalize import load_layout, normalize
from signlnk_ml.schemas_generated.landmark_layout_v1 import LandmarkLayoutV1


@dataclass(frozen=True)
class LandmarkStats:
    """Per-landmark statistics over all valid frames: [N, 3] mean/std and [N] presence."""

    mean: npt.NDArray[np.float64]
    std: npt.NDArray[np.float64]
    presence: npt.NDArray[np.float64]  # fraction of frames where the landmark is present
    frames: int


@dataclass(frozen=True)
class ParityReport:
    """Worst per-landmark difference between two LandmarkStats, in normalized units
    (1.0 = one shoulder width); `presence` is a fraction."""

    xy_mean: float
    xy_std: float
    z_mean: float
    z_std: float
    presence: float


def landmark_stats(sequences: Sequence[npt.ArrayLike]) -> LandmarkStats:
    """Normalizes each sequence, then takes NaN-aware statistics over all frames."""
    normalized = [normalize(s) for s in sequences]
    frames = np.concatenate(normalized, axis=0).astype(np.float64)
    frames = frames[np.isfinite(frames).any(axis=(1, 2))]  # drop frames normalization voided
    if len(frames) == 0:
        raise ValueError("no valid frames to compute statistics from")
    with warnings.catch_warnings():
        # A landmark that is never present gives NaN statistics on purpose.
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(frames, axis=0)
        std = np.nanstd(frames, axis=0)
    return LandmarkStats(
        mean=mean,
        std=std,
        presence=np.isfinite(frames[..., 0]).mean(axis=0),
        frames=len(frames),
    )


def _worst(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """Largest absolute per-landmark difference, ignoring landmarks missing on either side."""
    diff = np.abs(a - b)
    return float(np.nanmax(diff)) if np.isfinite(diff).any() else float("nan")


def parity_groups(layout: LandmarkLayoutV1 | None = None) -> dict[str, npt.NDArray[np.intp]]:
    """Landmark subsets that are comparable across signers.

    `face` is every face landmark; `head` is BlazePose nose, eyes and ears. Hands and arms are left
    out on purpose: which hand a signer uses and how they move it varies far more between people
    than between pipelines (see ADR-0006), so they are covered by the geometry checks instead.
    """
    layout = layout or load_layout()
    face = np.concatenate(
        [
            np.arange(s.start, s.stop)
            for g in layout.groups
            if g.name.startswith("face_")
            for s in [group_slice(layout, g.name)]
        ]
    )
    head = np.array([pose_position(layout, i) for i in (0, 2, 5, 7, 8)])
    return {"face": face.astype(np.intp), "head": head.astype(np.intp)}


def compare(
    train: LandmarkStats, serve: LandmarkStats, indices: npt.NDArray[np.intp] | None = None
) -> ParityReport:
    """Worst per-landmark difference over `indices` (default: all landmarks)."""
    pick = slice(None) if indices is None else indices
    return ParityReport(
        xy_mean=_worst(train.mean[pick, :2], serve.mean[pick, :2]),
        xy_std=_worst(train.std[pick, :2], serve.std[pick, :2]),
        z_mean=_worst(train.mean[pick, 2], serve.mean[pick, 2]),
        z_std=_worst(train.std[pick, 2], serve.std[pick, 2]),
        presence=_worst(train.presence[pick], serve.presence[pick]),
    )
