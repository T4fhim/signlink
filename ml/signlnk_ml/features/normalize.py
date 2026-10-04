"""Landmark normalization for slk-landmarks-v1.

Mirrors packages/landmarks/src/normalize.ts. Change both together; the golden fixtures in
tests/fixtures/normalization are the parity contract (max abs diff <= 1e-5).

Per frame: x and y are centred on the neck (midpoint of the two shoulder anchors) and divided by
the shoulder width (Euclidean distance in x, y). z is kept unchanged. A frame whose shoulders are
missing or closer than MIN_SHOULDER_WIDTH cannot be normalized and becomes all NaN. Math runs in
float64 and is cast to float32 at the end, in the same operation order as the TS version.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import numpy.typing as npt

from signlnk_ml.schemas_generated.landmark_layout_v1 import LandmarkLayoutV1

REPO_ROOT = Path(__file__).resolve().parents[3]
LAYOUT_PATH = REPO_ROOT / "packages" / "schemas" / "layouts" / "slk-landmarks-v1.json"

#: Shoulder widths below this are treated as degenerate (normalized image units).
MIN_SHOULDER_WIDTH = 1e-6


@cache
def load_layout(path: Path = LAYOUT_PATH) -> LandmarkLayoutV1:
    """Loads and validates a layout document (default: slk-landmarks-v1)."""
    return LandmarkLayoutV1.model_validate_json(path.read_text(encoding="utf-8"))


def shoulder_indices(layout: LandmarkLayoutV1) -> tuple[int, int]:
    """Output-layout indices of the left and right shoulder anchors."""
    anchors = layout.anchors or {}
    try:
        return anchors["left_shoulder"].root, anchors["right_shoulder"].root
    except KeyError as error:
        raise ValueError(f"layout {layout.layout_id} has no {error.args[0]} anchor") from error


def normalize(
    landmarks: npt.ArrayLike, layout: LandmarkLayoutV1 | None = None
) -> npt.NDArray[np.float32]:
    """Normalizes landmarks of shape [T, N, 3] (or a single frame [N, 3]) to float32.

    The input is not modified. NaN marks a missing landmark and stays NaN.
    """
    layout = layout or load_layout()
    array = np.asarray(landmarks, dtype=np.float64)
    if array.ndim == 2:
        frame: npt.NDArray[np.float32] = normalize(array[None], layout)[0]
        return frame
    if array.ndim != 3 or array.shape[1:] != (layout.n_landmarks, 3):
        raise ValueError(f"expected [T, {layout.n_landmarks}, 3], got {array.shape}")

    left_index, right_index = shoulder_indices(layout)
    left = array[:, left_index, :2]
    right = array[:, right_index, :2]
    center = (left + right) / 2.0
    delta = right - left
    width = np.sqrt(delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1])

    valid = np.isfinite(center).all(axis=1) & np.isfinite(width) & (width >= MIN_SHOULDER_WIDTH)
    safe_width = np.where(valid, width, 1.0)

    out = array.copy()
    out[..., :2] = (array[..., :2] - center[:, None, :]) / safe_width[:, None, None]
    out[~valid] = np.nan
    return out.astype(np.float32)
