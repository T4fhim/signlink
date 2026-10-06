"""Aspect conversion: hand-computed numbers and parity with the shared golden cases (TS mirror)."""

import json

import numpy as np
import numpy.typing as npt
import pytest
from signlnk_ml.features.aspect import aspect_scale, to_reference_aspect
from signlnk_ml.features.normalize import REPO_ROOT, load_layout

LAYOUT = load_layout()
N = LAYOUT.n_landmarks
CASES = json.loads((REPO_ROOT / "tests" / "fixtures" / "aspect" / "cases.json").read_text())
TOLERANCE = 1e-5


def test_layout_has_a_reference_aspect() -> None:
    assert LAYOUT.reference_aspect == 0.78


def test_scale_is_reference_aspect_over_image_aspect() -> None:
    assert aspect_scale(640, 480) == pytest.approx(0.585, abs=1e-12)  # 4:3 webcam
    assert aspect_scale(1000, 1000) == pytest.approx(0.78, abs=1e-12)  # square
    assert aspect_scale(480, 640) == pytest.approx(0.78 / 0.75, abs=1e-12)  # 3:4 portrait
    assert aspect_scale(780, 1000) == pytest.approx(1.0, abs=1e-12)  # already the reference


@pytest.mark.parametrize("size", [(0, 480), (640, -1), (float("nan"), 480)])
def test_invalid_sizes_are_rejected(size: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="invalid image size"):
        aspect_scale(*size)


def test_a_layout_without_reference_aspect_is_rejected() -> None:
    with pytest.raises(ValueError, match="reference_aspect"):
        aspect_scale(640, 480, LAYOUT.model_copy(update={"reference_aspect": None}))


def test_only_y_is_rescaled_and_nan_stays() -> None:
    padded = np.full((N, 3), np.nan)
    padded[:3] = [[0.5, 0.4, 0.1], [np.nan, 0.8, np.nan], [0.2, np.nan, 0.3]]
    out = to_reference_aspect(padded, 640, 480)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0], [0.5, 0.4 * 0.585, 0.1], atol=1e-6)
    assert np.isnan(out[1, 0]) and np.isnan(out[1, 2])
    np.testing.assert_allclose(out[1, 1], 0.8 * 0.585, atol=1e-6)
    assert np.isnan(out[2, 1])
    assert np.isnan(out[3:]).all()


def test_the_input_is_not_modified() -> None:
    a = np.full((N, 3), 0.5)
    to_reference_aspect(a, 640, 480)
    assert (a == 0.5).all()


def _to_array(values: list[float | None], shape: list[int]) -> npt.NDArray[np.float32]:
    flat = np.array([np.nan if v is None else v for v in values], dtype=np.float32)
    return flat.reshape(shape)


@pytest.mark.parametrize("case", CASES, ids=lambda c: str(c["name"]))
def test_matches_golden_case(case: dict[str, object]) -> None:
    shape = case["shape"]
    width, height = case["width"], case["height"]
    assert isinstance(shape, list) and isinstance(width, int) and isinstance(height, int)
    actual = to_reference_aspect(_to_array(case["input"], shape), width, height)  # type: ignore[arg-type]
    expected = _to_array(case["expected"], shape)  # type: ignore[arg-type]
    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(
        actual[~np.isnan(actual)], expected[~np.isnan(expected)], atol=TOLERANCE, rtol=0
    )
