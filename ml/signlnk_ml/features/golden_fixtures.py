"""Writes the golden fixtures for the normalization parity test.

Run: uv run python -m signlnk_ml.features.golden_fixtures

Each fixture is JSON {name, layout_id, shape: [T, N, 3], input, expected} with flat row-major
arrays and null for NaN. `expected` is produced by the Python reference implementation; the TS and
Python tests both check it, and hand-computed cases pin the maths independently.
"""

from __future__ import annotations

import json
import math

import numpy as np
import numpy.typing as npt

from signlnk_ml.features.normalize import REPO_ROOT, load_layout, normalize, shoulder_indices

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "normalization"
SEED = 20261004


def _blank(frames: int, n: int) -> npt.NDArray[np.float32]:
    return np.full((frames, n, 3), np.nan, dtype=np.float32)


def hand_computed(n: int, left: int, right: int) -> npt.NDArray[np.float32]:
    """One frame small enough to normalize by hand (see test_normalize.py for the numbers)."""
    a = _blank(1, n)
    a[0, left] = (0.6, 0.5, 0.0)
    a[0, right] = (0.4, 0.5, 0.0)
    a[0, 0] = (0.5, 0.3, 0.7)
    a[0, 1] = (0.7, 0.9, -0.4)
    a[0, 100] = (0.4, 0.5, 0.25)
    return a


def random_clip(n: int, left: int, right: int) -> npt.NDArray[np.float32]:
    """Eight random frames with realistic shoulders and several kinds of dropout."""
    rng = np.random.default_rng(SEED)
    frames = 8
    xy = rng.uniform(0.0, 1.0, size=(frames, n, 2))
    z = rng.normal(0.0, 0.1, size=(frames, n, 1))
    a = np.concatenate([xy, z], axis=2).astype(np.float32)
    jitter = rng.uniform(-0.02, 0.02, size=(frames, 4))
    a[:, left, :2] = np.stack([0.62 + jitter[:, 0], 0.60 + jitter[:, 1]], axis=1)
    a[:, right, :2] = np.stack([0.38 + jitter[:, 2], 0.60 + jitter[:, 3]], axis=1)
    a[2, :42] = np.nan  # both hands missing
    a[5, 53:] = np.nan  # face missing
    a[6, 42:53] = np.nan  # pose missing, so shoulders too: whole frame becomes NaN
    a[7, right, :2] = a[7, left, :2]  # zero shoulder width
    return a


def edge_cases(n: int, left: int, right: int) -> npt.NDArray[np.float32]:
    """Frames that exercise the NaN and degenerate-width policy."""
    a = _blank(4, n)
    for t in range(4):
        a[t, 0] = (0.55, 0.45, 0.1)
        a[t, 1] = (0.30, np.nan, 0.2)  # one coordinate missing
        a[t, 2] = (0.45, 0.65, np.nan)  # z missing
    a[0, left] = (np.nan, 0.5, 0.0)  # left shoulder x missing -> frame invalid
    a[0, right] = (0.4, 0.5, 0.0)
    a[1, left] = (0.5, 0.5, 0.0)  # zero width -> invalid
    a[1, right] = (0.5, 0.5, 0.0)
    a[2, left] = (0.5, 0.5, 0.0)  # width 5e-7 < MIN_SHOULDER_WIDTH -> invalid
    a[2, right] = (0.5 + 5e-7, 0.5, 0.0)
    a[3, left] = (0.7, 0.4, np.nan)  # shoulder z missing is fine: z is not used
    a[3, right] = (0.3, 0.6, 0.0)
    return a


def _flat(values: npt.NDArray[np.float32]) -> list[float | None]:
    return [None if math.isnan(v) else v for v in values.ravel().tolist()]


def build() -> dict[str, dict[str, object]]:
    layout = load_layout()
    left, right = shoulder_indices(layout)
    inputs = {
        "hand_computed": hand_computed(layout.n_landmarks, left, right),
        "random_clip": random_clip(layout.n_landmarks, left, right),
        "edge_cases": edge_cases(layout.n_landmarks, left, right),
    }
    return {
        name: {
            "name": name,
            "layout_id": layout.layout_id,
            "shape": list(array.shape),
            "input": _flat(array),
            "expected": _flat(normalize(array, layout)),
        }
        for name, array in inputs.items()
    }


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fixture in build().items():
        path = FIXTURE_DIR / f"{name}.json"
        path.write_text(json.dumps(fixture, separators=(",", ":"), allow_nan=False) + "\n")
        print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
