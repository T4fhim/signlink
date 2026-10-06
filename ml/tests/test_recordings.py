"""Recordings: the .npy written by the TS recorder loads in Python (golden file), validated."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from signlnk_ml.data.recordings import load_recording, recordings_dir
from signlnk_ml.features.normalize import REPO_ROOT, load_layout

GOLDEN = REPO_ROOT / "tests" / "fixtures" / "npy" / "pattern.npy"
N = load_layout().n_landmarks


def pattern(frames: int = 4) -> npt.NDArray[np.float32]:
    """Same array as packages/landmarks/src/npyPattern.ts."""
    t, k, c = np.meshgrid(np.arange(frames), np.arange(N), np.arange(3), indexing="ij")
    values = (t * 1000 + k + 0.25 * c).astype(np.float32)
    values[(t + k + c) % 17 == 0] = np.nan
    return values


def test_golden_npy_written_by_typescript_loads_in_python() -> None:
    loaded = load_recording(GOLDEN)
    expected = pattern()
    assert loaded.dtype == np.float32
    assert loaded.shape == (4, N, 3)
    np.testing.assert_array_equal(np.isnan(loaded), np.isnan(expected))
    np.testing.assert_array_equal(loaded[~np.isnan(loaded)], expected[~np.isnan(expected)])


def test_header_is_numpy_version_1_and_64_byte_aligned() -> None:
    raw = GOLDEN.read_bytes()
    assert raw[:8] == b"\x93NUMPY\x01\x00"
    header_length = int.from_bytes(raw[8:10], "little")
    assert (10 + header_length) % 64 == 0
    assert raw[10 : 10 + header_length].endswith(b"\n")
    assert len(raw) == 10 + header_length + 4 * 4 * N * 3


@pytest.mark.parametrize("shape", [(3, N + 1, 3), (3, N, 2), (N, 3), (0, N, 3)])
def test_wrong_shapes_are_rejected(tmp_path: Path, shape: tuple[int, ...]) -> None:
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros(shape, dtype=np.float32))
    with pytest.raises(ValueError, match="expected"):
        load_recording(path)


def test_float64_recordings_are_converted(tmp_path: Path) -> None:
    path = tmp_path / "f64.npy"
    np.save(path, np.zeros((2, N, 3), dtype=np.float64))
    assert load_recording(path).dtype == np.float32


def test_recordings_live_under_the_data_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SIGNLNK_DATA_DIR", str(tmp_path))
    assert recordings_dir() == tmp_path / "serve-recordings"
