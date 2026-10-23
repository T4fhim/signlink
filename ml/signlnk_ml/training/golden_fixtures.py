"""Writes the golden fixtures for the model-input parity test (TS `modelInput.ts` mirror).

Run: uv run python -m signlnk_ml.training.golden_fixtures

One JSON file, tests/fixtures/model_input/cases.json, {make_input: [...], resample: [...]}.
Inputs are normalized clips [T, N, 3] (flat row-major, null for NaN); `expected` is produced by
`features.py`. The TS and Python tests both check it.
"""

from __future__ import annotations

import json
import math

import numpy as np
import numpy.typing as npt

from signlnk_ml.features.golden_fixtures import edge_cases, random_clip
from signlnk_ml.features.normalize import REPO_ROOT, load_layout, normalize, shoulder_indices
from signlnk_ml.training.features import make_input, resample_nearest

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "model_input" / "cases.json"
RESAMPLE_CASES = [(8, 5), (8, 8), (3, 64), (8, 64)]  # (frames, length)
RESAMPLE_LANDMARKS = 4


def _flat(values: npt.NDArray[np.float32]) -> list[float | None]:
    return [None if math.isnan(v) else v for v in values.ravel().tolist()]


def build() -> dict[str, list[dict[str, object]]]:
    layout = load_layout()
    left, right = shoulder_indices(layout)
    n = layout.n_landmarks
    clips = {
        "random_clip": normalize(random_clip(n, left, right), layout),
        "edge_cases": normalize(edge_cases(n, left, right), layout),
    }
    inputs: list[dict[str, object]] = [
        {
            "name": f"{name}_{'xyz' if use_z else 'xy'}",
            "use_z": use_z,
            "shape": list(clip.shape),
            "input": _flat(clip),
            "expected": _flat(make_input(clip, use_z=use_z)),
        }
        for name, clip in clips.items()
        for use_z in (False, True)
    ]
    # Resampling is shape-agnostic: four landmarks (with the clip's NaN frames) keep the file small.
    head = clips["random_clip"][:, :RESAMPLE_LANDMARKS]
    resamples: list[dict[str, object]] = [
        {
            "name": f"{frames}_to_{length}",
            "length": length,
            "shape": [frames, RESAMPLE_LANDMARKS, 3],
            "input": _flat(head[:frames]),
            "expected": _flat(resample_nearest(head[:frames], length)),
        }
        for frames, length in RESAMPLE_CASES
    ]
    return {"make_input": inputs, "resample": resamples}


def main() -> None:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(build(), separators=(",", ":"), allow_nan=False)
    FIXTURE.write_text(text + "\n")
    print(f"wrote {FIXTURE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
