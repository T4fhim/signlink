"""Conversion of landmarks to the layout's reference aspect ratio (ADR-0007).

MediaPipe normalizes x by image width and y by image height, so the same face has different vertical
proportions on a 4:3 webcam and on the portrait phone video Kaggle ISLR was recorded on. Mirrors
packages/landmarks/src/aspect.ts; tests/fixtures/aspect/cases.json is the parity contract.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from signlnk_ml.features.normalize import load_layout
from signlnk_ml.schemas_generated.landmark_layout_v1 import LandmarkLayoutV1


def aspect_scale(width: float, height: float, layout: LandmarkLayoutV1 | None = None) -> float:
    """Multiplier for y that moves a `width x height` frame to the layout's reference aspect."""
    layout = layout or load_layout()
    if not (width > 0 and height > 0):
        raise ValueError(f"invalid image size {width}x{height}")
    if layout.reference_aspect is None:
        raise ValueError(f"layout {layout.layout_id} has no reference_aspect")
    return layout.reference_aspect / (width / height)


def to_reference_aspect(
    landmarks: npt.ArrayLike,
    width: float,
    height: float,
    layout: LandmarkLayoutV1 | None = None,
) -> npt.NDArray[np.float32]:
    """Rescales y of landmarks [..., N, 3] to the reference aspect; returns a new float32 array.

    x and z are unchanged and NaN stays NaN. Math runs in float64, cast to float32 at the end.
    """
    scale = aspect_scale(width, height, layout)
    out = np.array(landmarks, dtype=np.float64)
    out[..., 1] = out[..., 1] * scale
    return out.astype(np.float32)
