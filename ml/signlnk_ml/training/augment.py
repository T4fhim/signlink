"""Training augmentations on normalized landmark windows [T, N, 3] (never applied at inference).

Mirroring swaps the hand slots and the left/right pose pairs and negates x. Face landmarks have no
verified left/right pairing in `slk-landmarks-v1`, so a mirrored window's face is marked missing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from signlnk_ml.data.geometry import group_slice, pose_position
from signlnk_ml.schemas_generated.landmark_layout_v1 import LandmarkLayoutV1

Array = npt.NDArray[np.float32]

#: BlazePose left/right pairs present in the layout's pose group (nose is on the midline).
POSE_PAIRS = ((2, 5), (7, 8), (11, 12), (13, 14), (15, 16))


def _mirror_plan(layout: LandmarkLayoutV1) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    permutation = np.arange(layout.n_landmarks)
    left, right = group_slice(layout, "hand_left"), group_slice(layout, "hand_right")
    permutation[left], permutation[right] = permutation[right].copy(), permutation[left].copy()
    for a, b in POSE_PAIRS:
        i, j = pose_position(layout, a), pose_position(layout, b)
        permutation[i], permutation[j] = j, i
    face = [
        np.arange(layout.n_landmarks)[group_slice(layout, g.name)]
        for g in layout.groups
        if g.name.startswith("face")
    ]
    return permutation, np.concatenate(face)


def mirror(window: Array, layout: LandmarkLayoutV1) -> Array:
    permutation, face = _mirror_plan(layout)
    out: Array = window[:, permutation].copy()
    out[..., 0] *= -1
    out[:, face] = np.nan
    return out


def rotate(window: Array, angle: float) -> Array:
    """Rotates x, y about the origin (the neck after normalization) by `angle` radians."""
    cos, sin = np.cos(angle), np.sin(angle)
    out = window.copy()
    out[..., 0] = window[..., 0] * cos - window[..., 1] * sin
    out[..., 1] = window[..., 0] * sin + window[..., 1] * cos
    return out


def scale(window: Array, factor: float) -> Array:
    out: Array = (window * factor).astype(np.float32)
    return out


def time_stretch(window: Array, factor: float) -> Array:
    """factor > 1 slows the sign down, < 1 speeds it up; the window length stays the same."""
    frames = window.shape[0]
    source = np.floor(np.arange(frames) / factor).astype(int)
    valid = source < frames
    out = np.full_like(window, np.nan)
    out[valid] = window[source[valid]]
    return out


def frame_drop(window: Array, probability: float, rng: np.random.Generator) -> Array:
    out = window.copy()
    out[rng.random(window.shape[0]) < probability] = np.nan
    return out


@dataclass(frozen=True)
class AugmentConfig:
    mirror_probability: float = 0.5
    rotate_degrees: float = 15.0
    scale_range: tuple[float, float] = (0.9, 1.1)
    stretch_range: tuple[float, float] = (0.8, 1.25)
    drop_probability: float = 0.1


class Augmenter:
    """Random chain: mirror, rotate, scale, time-stretch, frame drop. Seeded, so reproducible."""

    def __init__(
        self, layout: LandmarkLayoutV1, seed: int, config: AugmentConfig | None = None
    ) -> None:
        self.layout = layout
        self.rng = np.random.default_rng(seed)
        self.config = config or AugmentConfig()

    def __call__(self, window: Array) -> Array:
        rng, config = self.rng, self.config
        if rng.random() < config.mirror_probability:
            window = mirror(window, self.layout)
        angle = rng.uniform(-config.rotate_degrees, config.rotate_degrees)
        window = rotate(window, np.deg2rad(angle))
        window = scale(window, rng.uniform(*config.scale_range))
        window = time_stretch(window, rng.uniform(*config.stretch_range))
        return frame_drop(window, config.drop_probability, rng).astype(np.float32)
