"""Shared fixtures. Tests that need real data skip cleanly when it is not on this machine (CI)."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from signlnk_ml.data.fetch_islr import data_dir
from signlnk_ml.data.kaggle_islr import load_slk_sequence

Sequences = dict[str, list[npt.NDArray[np.float32]]]


@pytest.fixture(scope="session")
def islr_root() -> Path:
    root = data_dir()
    if not (root / "sample_manifest.csv").exists():
        pytest.skip(
            "Kaggle ISLR sample not found: set SIGNLNK_DATA_DIR and run "
            "`uv run python -m signlnk_ml.data.fetch_islr`"
        )
    return root


@pytest.fixture(scope="session")
def islr_sample(islr_root: Path) -> Sequences:
    """participant_id -> sequences mapped to slk-landmarks-v1 (raw, un-normalized)."""
    sequences: Sequences = defaultdict(list)
    with (islr_root / "sample_manifest.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sequences[row["participant_id"]].append(load_slk_sequence(islr_root / row["path"]))
    return dict(sequences)
