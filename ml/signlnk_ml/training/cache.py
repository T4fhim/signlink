"""Fixed-length windows from Kaggle ISLR, cached per split (Phase 1 step 3).

Each sign sequence becomes one normalized [length, N, 3] window (float16 on disk). The 'other' class
(no sign) is cut from runs where neither hand is detected, at most one per sequence and capped at
`other_ratio` times the average sign class size, always from the same signer's split.

    uv run python -m signlnk_ml.training.cache --config ml/configs/baseline.yaml --out-dir <dir>
        [--data-dir <kaggle-islr dir>] [--sample] [--workers 4]

`--sample` reads sample_manifest.csv (the 525-sequence local sample) instead of the full train.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from signlnk_ml.data.fetch_islr import data_dir
from signlnk_ml.data.kaggle_islr import load_sign_map, read_sequence, to_slk_v1
from signlnk_ml.data.splits import SPLIT_NAMES, load_assignment
from signlnk_ml.features.normalize import REPO_ROOT, load_layout, normalize
from signlnk_ml.schemas_generated.landmark_layout_v1 import LandmarkLayoutV1
from signlnk_ml.training.config import TrainConfig, load_config
from signlnk_ml.training.features import hands_absent_runs, resample_nearest

Array = npt.NDArray[np.float32]


def sequence_to_window(legacy: Array, layout: LandmarkLayoutV1, length: int) -> Array:
    """Legacy Holistic tensor [T, 543, 3] -> normalized slk-landmarks-v1 window [length, N, 3]."""
    if length < 1:
        raise ValueError("window length must be positive")
    return resample_nearest(normalize(to_slk_v1(legacy, layout), layout), length)


def other_windows(
    raw: Array, layout: LandmarkLayoutV1, min_run: int, length: int, max_windows: int
) -> list[Array]:
    """'No sign' windows from the longest hands-absent runs of a raw (un-normalized) sequence."""
    runs = sorted(hands_absent_runs(raw, layout, min_run), key=lambda r: (r[0] - r[1], r[0]))
    return [
        resample_nearest(normalize(raw[start:end], layout), length)
        for start, end in runs[:max_windows]
    ]


def cap_other(available: int, n_sign_windows: int, n_sign_classes: int, ratio: float) -> int:
    """At most `ratio` times the average sign class size."""
    return min(available, round(ratio * n_sign_windows / n_sign_classes))


def select_indices(n: int, k: int, seed: int) -> npt.NDArray[np.intp]:
    """A sorted, seeded random subset of range(n) of size min(k, n)."""
    if k >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=k, replace=False))


def _process(
    root: Path, row: Mapping[str, str], layout: LandmarkLayoutV1, cfg: TrainConfig
) -> tuple[Array, list[Array]]:
    legacy = read_sequence(root / row["path"])
    window = sequence_to_window(legacy, layout, cfg.window.length)
    others = other_windows(
        to_slk_v1(legacy, layout), layout, cfg.window.other_min_run, cfg.window.length, 1
    )
    return window, others


def build_split_cache(  # noqa: PLR0913, PLR0917
    root: Path,
    rows: list[dict[str, str]],
    split: str,
    out_dir: Path,
    cfg: TrainConfig,
    workers: int = 4,
) -> dict[str, Any]:
    layout = load_layout()
    sign_map = load_sign_map(root)
    other_index = len(sign_map)
    n, n_signs = len(rows), len(sign_map)
    cap = cap_other(n, n, n_signs, cfg.window.other_ratio)  # upper bound: one candidate per row
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = np.lib.format.open_memmap(
        out_dir / "windows.npy",
        mode="w+",
        dtype=np.float16,
        shape=(n + cap, cfg.window.length, layout.n_landmarks, 3),
    )
    labels = np.full(n + cap, -1, dtype=np.int16)
    participants = np.full(n + cap, "", dtype="<U12")
    candidates: list[tuple[Array, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda r: _process(root, r, layout, cfg), rows)
        for i, (row, (window, others)) in enumerate(zip(rows, results, strict=True)):
            windows[i] = window.astype(np.float16)
            labels[i], participants[i] = sign_map[row["sign"]], row["participant_id"]
            candidates.extend((other, row["participant_id"]) for other in others)
    wanted = cap_other(len(candidates), n, n_signs, cfg.window.other_ratio)
    keep = select_indices(len(candidates), wanted, seed=cfg.train.seed)
    for j, index in enumerate(keep):
        windows[n + j] = candidates[index][0].astype(np.float16)
        labels[n + j], participants[n + j] = other_index, candidates[index][1]
    total = n + len(keep)
    windows.flush()
    np.save(out_dir / "labels.npy", labels[:total])
    np.save(out_dir / "participants.npy", participants[:total])
    meta = {
        "split": split,
        "n": total,
        "n_signs": n,
        "n_other": len(keep),
        "length": cfg.window.length,
        "n_classes": n_signs + 1,
        "other_index": other_index,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def load_split(
    cache_dir: Path, split: str
) -> tuple[npt.NDArray[np.float16], npt.NDArray[np.int16], npt.NDArray[np.str_], dict[str, Any]]:
    """(windows, labels, participants, meta); windows are memory-mapped, not loaded into RAM."""
    folder = cache_dir / split
    meta: dict[str, Any] = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    windows = np.load(folder / "windows.npy", mmap_mode="r")[: meta["n"]]
    return windows, np.load(folder / "labels.npy"), np.load(folder / "participants.npy"), meta


def read_rows(root: Path, sample: bool) -> list[dict[str, str]]:
    name = "sample_manifest.csv" if sample else "train.csv"
    with (root / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    default_config = REPO_ROOT / "ml" / "configs" / "baseline.yaml"
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = args.data_dir or data_dir()
    assignment = load_assignment(REPO_ROOT / cfg.splits)
    rows = read_rows(root, args.sample)
    for split in SPLIT_NAMES:
        mine = [r for r in rows if assignment[r["participant_id"]] == split]
        meta = build_split_cache(root, mine, split, args.out_dir / split, cfg, args.workers)
        print(f"{split}: {meta['n_signs']} sign windows + {meta['n_other']} 'other' windows")


if __name__ == "__main__":
    main()
