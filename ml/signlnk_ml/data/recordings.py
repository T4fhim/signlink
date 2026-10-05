"""Browser recordings in slk-landmarks-v1: `.npy` float32 [T, N, 3], raw, NaN = missing.

Recordings contain face landmarks, which can identify a person, so they stay on this machine under
$SIGNLNK_DATA_DIR/serve-recordings (gitignored) and are never committed or uploaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from signlnk_ml.data.fetch_islr import data_root
from signlnk_ml.features.normalize import load_layout
from signlnk_ml.schemas_generated.landmark_layout_v1 import LandmarkLayoutV1


def recordings_dir() -> Path:
    return data_root() / "serve-recordings"


def load_recording(path: Path, layout: LandmarkLayoutV1 | None = None) -> npt.NDArray[np.float32]:
    """Loads one recording and checks it matches the layout."""
    layout = layout or load_layout()
    array = np.load(path, allow_pickle=False)
    if array.ndim != 3 or array.shape[0] < 1 or array.shape[1:] != (layout.n_landmarks, 3):
        raise ValueError(f"{path.name}: expected [T, {layout.n_landmarks}, 3], got {array.shape}")
    return np.asarray(array, dtype=np.float32)
