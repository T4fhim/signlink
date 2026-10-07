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


def make_input(window: Array, use_z: bool) -> Array:
    """[T, N, 3] -> [T, N * d + N]: coordinates (NaN -> 0) then a 1/0 'landmark present' mask."""
    coords = window if use_z else window[..., :2]
    present = np.isfinite(coords).all(axis=-1)
    clean = np.where(np.isfinite(coords), coords, 0.0).astype(np.float32)
    frames = window.shape[0]
    out: Array = np.concatenate([clean.reshape(frames, -1), present.astype(np.float32)], axis=1)
    return out
