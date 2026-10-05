"""Kaggle ISLR loader on synthetic parquet files in the real long-form schema."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from signlnk_ml.data.fetch_islr import SequenceRow, select_sample
from signlnk_ml.data.kaggle_islr import (
    LEGACY_COUNTS,
    LEGACY_OFFSETS,
    LEGACY_TOTAL,
    legacy_columns,
    load_sign_map,
    read_sequence,
    to_slk_v1,
)
from signlnk_ml.features.normalize import load_layout

LAYOUT = load_layout()
N = LAYOUT.n_landmarks


def write_parquet(
    path: Path, frames: list[int], *, drop: set[tuple[str, int]] | None = None
) -> None:
    """Long-form file where x encodes the legacy column, y the frame, z a constant."""
    rows: dict[str, list[object]] = {
        k: [] for k in ("frame", "row_id", "type", "landmark_index", "x", "y", "z")
    }
    for frame in frames:
        for legacy_type, count in LEGACY_COUNTS.items():
            for index in range(count):
                missing = (legacy_type, index) in (drop or set())
                rows["frame"].append(frame)
                rows["row_id"].append(f"{frame}-{legacy_type}-{index}")
                rows["type"].append(legacy_type)
                rows["landmark_index"].append(index)
                column = float(LEGACY_OFFSETS[legacy_type] + index)
                rows["x"].append(float("nan") if missing else column)
                rows["y"].append(float(frame))
                rows["z"].append(0.5)
    schema = pa.schema(
        [
            ("frame", pa.int16()),
            ("row_id", pa.string()),
            ("type", pa.string()),
            ("landmark_index", pa.int16()),
            ("x", pa.float64()),
            ("y", pa.float64()),
            ("z", pa.float64()),
        ]
    )
    pq.write_table(pa.Table.from_pydict(rows, schema=schema), path)


def single_row_table(legacy_type: str, index: int) -> pa.Table:
    return pa.table(
        {
            "frame": pa.array([0], pa.int16()),
            "row_id": [f"0-{legacy_type}-{index}"],
            "type": [legacy_type],
            "landmark_index": pa.array([index], pa.int16()),
            "x": [0.1],
            "y": [0.1],
            "z": [0.1],
        }
    )


def test_read_sequence_builds_the_dense_legacy_tensor(tmp_path: Path) -> None:
    path = tmp_path / "seq.parquet"
    write_parquet(path, [20, 21, 22])
    legacy = read_sequence(path)
    assert legacy.shape == (3, LEGACY_TOTAL, 3)
    assert legacy.dtype == np.float32
    assert LEGACY_TOTAL == 543
    np.testing.assert_array_equal(legacy[0, :, 0], np.arange(LEGACY_TOTAL))  # x = column
    np.testing.assert_array_equal(legacy[:, 7, 1], [20, 21, 22])  # y = frame, in time order


def test_missing_rows_stay_nan(tmp_path: Path) -> None:
    path = tmp_path / "seq.parquet"
    write_parquet(path, [0, 1], drop={("left_hand", 3), ("pose", 0)})
    legacy = read_sequence(path)
    assert np.isnan(legacy[:, LEGACY_OFFSETS["left_hand"] + 3, 0]).all()
    assert np.isnan(legacy[:, LEGACY_OFFSETS["pose"], 0]).all()
    assert np.isfinite(legacy[:, LEGACY_OFFSETS["left_hand"] + 4, 0]).all()


def test_unknown_type_is_rejected(tmp_path: Path) -> None:
    pq.write_table(single_row_table("eyes", 0), tmp_path / "unknown.parquet")
    with pytest.raises(ValueError, match="unknown landmark type"):
        read_sequence(tmp_path / "unknown.parquet")


def test_out_of_range_index_is_rejected(tmp_path: Path) -> None:
    pq.write_table(single_row_table("pose", 40), tmp_path / "bad.parquet")
    with pytest.raises(ValueError, match="out of range"):
        read_sequence(tmp_path / "bad.parquet")


def test_to_slk_v1_picks_the_layouts_landmarks() -> None:
    legacy = np.zeros((2, LEGACY_TOTAL, 3), dtype=np.float32)
    legacy[..., 0] = np.arange(LEGACY_TOTAL)  # x = legacy column
    out = to_slk_v1(legacy)
    assert out.shape == (2, N, 3)
    columns = legacy_columns(LAYOUT)
    assert None not in columns
    np.testing.assert_array_equal(out[0, :, 0], np.array(columns, dtype=np.float32))
    # spot checks by hand: layout starts with the left hand, then right hand, then pose
    assert out[0, 0, 0] == LEGACY_OFFSETS["left_hand"]
    assert out[0, 21, 0] == LEGACY_OFFSETS["right_hand"]
    assert out[0, 42, 0] == LEGACY_OFFSETS["pose"]  # nose
    left_shoulder = (LAYOUT.anchors or {})["left_shoulder"].root
    assert out[0, left_shoulder, 0] == LEGACY_OFFSETS["pose"] + 11


def test_to_slk_v1_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="expected"):
        to_slk_v1(np.zeros((2, 100, 3)))


def test_layout_without_legacy_indices_is_rejected() -> None:
    first = LAYOUT.groups[0].model_copy(update={"legacy_holistic_indices": None})
    broken = LAYOUT.model_copy(update={"groups": [first, *LAYOUT.groups[1:]]})
    with pytest.raises(ValueError, match="legacy_holistic_indices"):
        legacy_columns(broken)


def test_a_none_legacy_index_becomes_nan() -> None:
    first = LAYOUT.groups[0]
    legacy_indices = [None, *[int(i.root) for i in first.indices][1:]]
    patched = first.model_copy(update={"legacy_holistic_indices": legacy_indices})
    layout = LAYOUT.model_copy(update={"groups": [patched, *LAYOUT.groups[1:]]})
    out = to_slk_v1(np.ones((1, LEGACY_TOTAL, 3), dtype=np.float32), layout)
    assert np.isnan(out[0, 0]).all()
    assert np.isfinite(out[0, 1]).all()


def test_load_sign_map(tmp_path: Path) -> None:
    (tmp_path / "sign_to_prediction_index_map.json").write_text(json.dumps({"TV": 0, "after": 1}))
    assert load_sign_map(tmp_path) == {"TV": 0, "after": 1}


def test_select_sample_is_deterministic_and_spread_over_signs() -> None:
    rows = [
        SequenceRow(f"p{p}/{s}.parquet", str(p), f"{p}{i:03d}{s}", s)
        for p in (1, 2)
        for i, s in enumerate(["a", "b", "c", "d"] * 5)
    ]
    chosen = select_sample(rows, 4)
    assert chosen == select_sample(rows, 4)
    assert len(chosen) == 8
    for participant in ("1", "2"):
        signs = [r.sign for r in chosen if r.participant_id == participant]
        assert sorted(set(signs)) == ["a", "b", "c", "d"]
    assert len(select_sample(rows, 1000)) == len(rows)
