"""Kaggle ISLR (Google asl-signs, legacy MediaPipe Holistic) -> slk-landmarks-v1.

Each parquet is long-form: columns frame, row_id, type, landmark_index, x, y, z with 543 rows per
frame (face 468, left_hand 21, pose 33, right_hand 21); missing parts are NaN rows. `read_sequence`
restores the dense legacy tensor [T, 543, 3]; `to_slk_v1` picks the layout's landmarks from it using
the layout's `legacy_holistic_indices`, so the layout file is the single mapping source.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from signlnk_ml.features.normalize import load_layout
from signlnk_ml.schemas_generated.landmark_layout_v1 import LandmarkLayoutV1

#: Row order of the legacy tensor: Kaggle's `type` values in alphabetical order.
LEGACY_COUNTS: dict[str, int] = {"face": 468, "left_hand": 21, "pose": 33, "right_hand": 21}
LEGACY_TOTAL = sum(LEGACY_COUNTS.values())  # 543
LEGACY_OFFSETS: dict[str, int] = {
    legacy_type: sum(list(LEGACY_COUNTS.values())[:position])
    for position, legacy_type in enumerate(LEGACY_COUNTS)
}

#: Layout group source -> Kaggle `type`. Verified against pose wrists on real data (ADR-0006).
SOURCE_TO_LEGACY_TYPE: dict[str, str] = {
    "hand_left": "left_hand",
    "hand_right": "right_hand",
    "pose": "pose",
    "face": "face",
}

Array = npt.NDArray[np.float32]


def read_sequence(path: Path) -> Array:
    """Reads one parquet into the dense legacy tensor [T, 543, 3] (float32, NaN = missing)."""
    table = pq.read_table(path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    frames = table["frame"].to_numpy()
    types = table["type"].to_numpy(zero_copy_only=False)
    index = table["landmark_index"].to_numpy()
    xyz = np.stack([table[c].to_numpy() for c in ("x", "y", "z")], axis=1)

    unique_frames, frame_pos = np.unique(frames, return_inverse=True)
    out = np.full((len(unique_frames), LEGACY_TOTAL, 3), np.nan, dtype=np.float32)
    known = np.zeros(len(types), dtype=bool)
    for legacy_type, offset in LEGACY_OFFSETS.items():
        mask = types == legacy_type
        if mask.any() and (index[mask].min() < 0 or index[mask].max() >= LEGACY_COUNTS[legacy_type]):
            raise ValueError(f"{path.name}: {legacy_type} landmark_index out of range")
        out[frame_pos[mask], offset + index[mask]] = xyz[mask]
        known |= mask
    if not known.all():
        raise ValueError(f"{path.name}: unknown landmark type {set(types[~known])}")
    return out


def _as_int(value: object) -> int | None:
    """Unwraps the generated RootModel wrappers (and None) around layout indices."""
    root = cast(Any, getattr(value, "root", value))
    return None if root is None else int(root)


def legacy_columns(layout: LandmarkLayoutV1) -> list[int | None]:
    """For each output landmark, its column in the legacy tensor (None = no counterpart)."""
    columns: list[int | None] = []
    for group in layout.groups:
        if group.legacy_holistic_indices is None:
            raise ValueError(f"layout group {group.name} has no legacy_holistic_indices")
        offset = LEGACY_OFFSETS[SOURCE_TO_LEGACY_TYPE[group.source.value]]
        legacy = [_as_int(v) for v in group.legacy_holistic_indices]
        if len(legacy) != len(group.indices):
            raise ValueError(f"layout group {group.name}: legacy and Tasks indices differ in length")
        columns.extend(None if v is None else offset + v for v in legacy)
    return columns


def to_slk_v1(legacy: npt.ArrayLike, layout: LandmarkLayoutV1 | None = None) -> Array:
    """Maps a legacy tensor [T, 543, 3] to slk-landmarks-v1 [T, N, 3] (still un-normalized)."""
    layout = layout or load_layout()
    array = np.asarray(legacy, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (LEGACY_TOTAL, 3):
        raise ValueError(f"expected [T, {LEGACY_TOTAL}, 3], got {array.shape}")
    out = np.full((array.shape[0], layout.n_landmarks, 3), np.nan, dtype=np.float32)
    for target, source in enumerate(legacy_columns(layout)):
        if source is not None:
            out[:, target] = array[:, source]
    return out


def load_slk_sequence(path: Path, layout: LandmarkLayoutV1 | None = None) -> Array:
    return to_slk_v1(read_sequence(path), layout)


def load_sign_map(root: Path) -> dict[str, int]:
    """sign name -> prediction index, from sign_to_prediction_index_map.json."""
    data: dict[str, int] = json.loads((root / "sign_to_prediction_index_map.json").read_text())
    return data


def sequences_of(paths: Sequence[Path], layout: LandmarkLayoutV1 | None = None) -> list[Array]:
    return [load_slk_sequence(p, layout) for p in paths]
