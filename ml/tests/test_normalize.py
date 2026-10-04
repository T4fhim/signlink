"""Normalization: golden fixtures (shared with the TS tests), hand-computed numbers, invariances."""

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from signlnk_ml.features.golden_fixtures import FIXTURE_DIR
from signlnk_ml.features.normalize import (
    MIN_SHOULDER_WIDTH,
    load_layout,
    normalize,
    shoulder_indices,
)

TOLERANCE = 1e-5  # max abs diff, the TS/Python parity budget
LAYOUT = load_layout()
N = LAYOUT.n_landmarks
LEFT, RIGHT = shoulder_indices(LAYOUT)
FIXTURES = sorted(FIXTURE_DIR.glob("*.json"))

Array = npt.NDArray[np.float32]


def _array(values: list[float | None], shape: list[int]) -> Array:
    flat = np.array([np.nan if v is None else v for v in values], dtype=np.float32)
    return flat.reshape(shape)


def assert_close(actual: Array, expected: Array) -> None:
    assert actual.shape == expected.shape
    missing = np.isnan(expected)
    assert np.array_equal(np.isnan(actual), missing), "NaN positions differ"
    np.testing.assert_allclose(actual[~missing], expected[~missing], atol=TOLERANCE, rtol=0)


def _clean_frames(frames: int = 5, seed: int = 7) -> Array:
    """Fully finite landmarks with plausible, non-degenerate shoulders."""
    rng = np.random.default_rng(seed)
    a: Array = rng.uniform(0.0, 1.0, size=(frames, N, 3)).astype(np.float32)
    a[:, LEFT, :2] = (0.65, 0.55)
    a[:, RIGHT, :2] = (0.35, 0.60)
    return a


def test_fixture_set_is_complete() -> None:
    assert [p.stem for p in FIXTURES] == ["edge_cases", "hand_computed", "random_clip"]


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_matches_golden_fixture(path: Path) -> None:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    assert fixture["layout_id"] == LAYOUT.layout_id
    shape = fixture["shape"]
    assert shape[1:] == [N, 3]
    assert_close(normalize(_array(fixture["input"], shape)), _array(fixture["expected"], shape))


def test_hand_computed_frame() -> None:
    """Shoulders (0.6, 0.5), (0.4, 0.5): centre (0.5, 0.5), width 0.2: x, y -> (p - 0.5) / 0.2."""
    a = np.full((N, 3), np.nan, dtype=np.float32)
    a[LEFT] = (0.6, 0.5, 0.0)
    a[RIGHT] = (0.4, 0.5, 0.0)
    a[0] = (0.5, 0.3, 0.7)
    a[1] = (0.7, 0.9, -0.4)
    a[100] = (0.4, 0.5, 0.25)

    expected = np.full((N, 3), np.nan, dtype=np.float32)
    expected[LEFT] = (0.5, 0.0, 0.0)
    expected[RIGHT] = (-0.5, 0.0, 0.0)
    expected[0] = (0.0, -1.0, 0.7)  # z is kept, not scaled
    expected[1] = (1.0, 2.0, -0.4)
    expected[100] = (-0.5, 0.0, 0.25)
    assert_close(normalize(a), expected)


def test_translation_invariant() -> None:
    a = _clean_frames().astype(np.float64)
    shifted = a.copy()
    shifted[..., 0] += 0.13
    shifted[..., 1] -= 0.21
    assert_close(normalize(shifted), normalize(a))


def test_uniform_scale_invariant() -> None:
    a = _clean_frames().astype(np.float64)
    scaled = a.copy()
    scaled[..., :2] *= 2.5
    assert_close(normalize(scaled), normalize(a))


def test_z_is_kept() -> None:
    a = _clean_frames()
    assert np.array_equal(normalize(a)[..., 2], a[..., 2])


def test_shoulders_land_one_unit_apart_around_the_origin() -> None:
    out = normalize(_clean_frames())
    left, right = out[:, LEFT, :2], out[:, RIGHT, :2]
    np.testing.assert_allclose((left + right) / 2, 0.0, atol=TOLERANCE)
    np.testing.assert_allclose(np.linalg.norm(left - right, axis=1), 1.0, atol=TOLERANCE)


def test_missing_landmarks_stay_missing_and_others_stay_finite() -> None:
    a = _clean_frames()
    a[1, 3, :] = np.nan
    a[2, 5, 1] = np.nan
    out = normalize(a)
    assert np.isnan(out[1, 3]).all()
    assert np.isnan(out[2, 5, 1]) and np.isfinite(out[2, 5, [0, 2]]).all()
    assert np.isfinite(out[0]).all()


@pytest.mark.parametrize("anchor", [LEFT, RIGHT])
def test_frame_without_a_shoulder_becomes_all_nan(anchor: int) -> None:
    a = _clean_frames()
    a[1, anchor, 0] = np.nan
    out = normalize(a)
    assert np.isnan(out[1]).all()
    assert np.isfinite(out[0]).all() and np.isfinite(out[2]).all()


def test_degenerate_width_threshold() -> None:
    a = _clean_frames(frames=2)
    a[:, RIGHT, :2] = a[:, LEFT, :2]
    a[0, RIGHT, 0] += np.float32(MIN_SHOULDER_WIDTH * 0.5)
    a[1, RIGHT, 0] += np.float32(MIN_SHOULDER_WIDTH * 4)
    out = normalize(a)
    assert np.isnan(out[0]).all()
    assert np.isfinite(out[1]).all()


def test_single_frame_matches_a_clip_of_one() -> None:
    a = _clean_frames(frames=1)
    assert np.array_equal(normalize(a[0]), normalize(a)[0])


def test_returns_float32_and_does_not_modify_the_input() -> None:
    a = _clean_frames()
    before = a.copy()
    out = normalize(a)
    assert out.dtype == np.float32
    assert np.array_equal(a, before)


@pytest.mark.parametrize("shape", [(N,), (4, N + 1, 3), (4, N, 2), (2, 4, N, 3)])
def test_wrong_shape_is_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="expected"):
        normalize(np.zeros(shape, dtype=np.float32))
