"""Fixed-length windows and model inputs from normalized landmark sequences [T, N, 3]."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from signlnk_ml.data.geometry import group_slice
from signlnk_ml.schemas_generated.landmark_layout_v1 import LandmarkLayoutV1

Array = npt.NDArray[np.float32]


def resample_nearest(window: Array, length: int) -> Array:
    """Nearest-frame resample to `length` frames. Missing values are never interpolated."""
    frames = window.shape[0]
    if frames == 0:
        raise ValueError("cannot resample an empty sequence")
    index = np.minimum((np.arange(length) * frames) // length, frames - 1)
    out: Array = window[index]
    return out


def hands_absent_runs(
    sequence: Array, layout: LandmarkLayoutV1, min_run: int
) -> list[tuple[int, int]]:
    """Maximal [start, end) frame runs of at least `min_run` frames with neither hand detected."""
    left, right = group_slice(layout, "hand_left"), group_slice(layout, "hand_right")
    present = np.isfinite(sequence[:, left, 0]).any(axis=1) | np.isfinite(
        sequence[:, right, 0]
    ).any(axis=1)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    flags: list[bool] = np.asarray(present).tolist()
    for frame, here in enumerate([*flags, True]):  # the sentinel closes a trailing run
        if not here and start is None:
            start = frame
        elif here and start is not None:
            if frame - start >= min_run:
                runs.append((start, frame))
            start = None
    return runs


def input_dim(n_landmarks: int, use_z: bool, velocity: bool = False) -> int:
    """Width of `make_input`'s rows."""
    coords = n_landmarks * (3 if use_z else 2)
    return coords + n_landmarks + (coords if velocity else 0)


def make_input(window: Array, use_z: bool, velocity: bool = False) -> Array:
    """[T, N, 3] -> [T, F]: coordinates (NaN -> 0), a 1/0 'landmark present' mask per landmark and,
    with `velocity`, frame-to-frame differences (0 where either frame lacks the landmark)."""
    coords = window if use_z else window[..., :2]
    present = np.asarray(np.isfinite(coords).all(axis=-1), dtype=bool)
    clean = np.where(np.isfinite(coords), coords, 0.0).astype(np.float32)
    frames = window.shape[0]
    parts = [clean.reshape(frames, -1), present.astype(np.float32)]
    if velocity:
        delta = np.zeros_like(clean)
        both = (present[1:] & present[:-1])[..., None]
        delta[1:] = (clean[1:] - clean[:-1]) * both
        parts.append(delta.reshape(frames, -1))
    out: Array = np.concatenate(parts, axis=1)
    return out
