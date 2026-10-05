"""Geometry checks and train/serve parity statistics on synthetic landmarks."""

import numpy as np
import numpy.typing as npt
import pytest
from signlnk_ml.data.geometry import (
    POSE_LEFT_EYE,
    POSE_LEFT_WRIST,
    POSE_RIGHT_EYE,
    POSE_RIGHT_WRIST,
    geometry_checks,
    group_slice,
    pose_position,
)
from signlnk_ml.data.parity import compare, landmark_stats
from signlnk_ml.features.normalize import load_layout, shoulder_indices

LAYOUT = load_layout()
N = LAYOUT.n_landmarks
LEFT_SHOULDER, RIGHT_SHOULDER = shoulder_indices(LAYOUT)

Array = npt.NDArray[np.float32]


def upright_person(frames: int = 6, seed: int = 3) -> Array:
    """A plausible unmirrored upright subject: subject's left is on the image's right (larger x)."""
    rng = np.random.default_rng(seed)
    a = np.full((frames, N, 3), np.nan, dtype=np.float32)

    def put(name: str, x: float, y: float, spread: float = 0.0) -> None:
        sl = group_slice(LAYOUT, name)
        count = sl.stop - sl.start
        base = np.array([x, y, 0.0], dtype=np.float32)
        wobble = rng.normal(0, 0.003, size=(frames, 1, 3)).astype(np.float32)
        a[:, sl] = base + wobble + rng.normal(0, spread, size=(frames, count, 3))

    put("hand_left", 0.72, 0.80, 0.02)
    put("hand_right", 0.28, 0.80, 0.02)
    put("face_nose_tip", 0.50, 0.40)
    put("face_lips", 0.50, 0.46, 0.01)
    put("face_left_eye", 0.54, 0.34, 0.005)
    put("face_right_eye", 0.46, 0.34, 0.005)
    put("face_left_eyebrow", 0.55, 0.30, 0.005)
    put("face_right_eyebrow", 0.45, 0.30, 0.005)
    # exact positions for the landmarks the checks compare against
    for index, (x, y) in {
        POSE_LEFT_WRIST: (0.72, 0.80),
        POSE_RIGHT_WRIST: (0.28, 0.80),
        POSE_LEFT_EYE: (0.54, 0.34),
        POSE_RIGHT_EYE: (0.46, 0.34),
        0: (0.50, 0.40),
        11: (0.65, 0.60),
        12: (0.35, 0.60),
    }.items():
        a[:, pose_position(LAYOUT, index), :] = (x, y, 0.0)
    # the hands' wrists (first landmark of each hand) sit exactly on the pose wrists
    a[:, group_slice(LAYOUT, "hand_left").start, :2] = (0.72, 0.80)
    a[:, group_slice(LAYOUT, "hand_right").start, :2] = (0.28, 0.80)
    return a


def test_anchors_point_at_the_pose_shoulders() -> None:
    assert pose_position(LAYOUT, 11) == LEFT_SHOULDER
    assert pose_position(LAYOUT, 12) == RIGHT_SHOULDER


def test_a_correct_subject_passes_every_check() -> None:
    checks = geometry_checks(upright_person())
    assert set(checks.values()) == {1.0}, checks


def test_swapped_hands_are_detected() -> None:
    a = upright_person()
    left, right = group_slice(LAYOUT, "hand_left"), group_slice(LAYOUT, "hand_right")
    a[:, left], a[:, right] = a[:, right].copy(), a[:, left].copy()
    checks = geometry_checks(a)
    assert checks["left_hand_at_left_wrist"] == 0.0
    assert checks["right_hand_at_right_wrist"] == 0.0
    assert checks["lips_below_nose"] == 1.0


def test_mirrored_image_is_detected() -> None:
    a = upright_person()
    a[..., 0] = 1.0 - a[..., 0]
    assert geometry_checks(a)["subject_left_is_image_right"] == 0.0


def test_upside_down_face_is_detected() -> None:
    a = upright_person()
    a[..., 1] = 1.0 - a[..., 1]
    checks = geometry_checks(a)
    assert checks["lips_below_nose"] == 0.0
    assert checks["left_brow_above_left_eye"] == 0.0


def test_missing_parts_give_nan_not_a_pass() -> None:
    a = upright_person()
    a[:, group_slice(LAYOUT, "hand_left")] = np.nan
    checks = geometry_checks(a)
    assert np.isnan(checks["left_hand_at_left_wrist"])
    assert checks["right_hand_at_right_wrist"] == 1.0


def test_identical_distributions_have_zero_difference() -> None:
    stats = landmark_stats([upright_person(seed=1)])
    report = compare(stats, stats)
    assert (report.xy_mean, report.xy_std, report.z_mean, report.z_std, report.presence) == (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def test_a_shifted_landmark_shows_up_in_the_mean() -> None:
    base = upright_person()
    moved = base.copy()
    moved[:, group_slice(LAYOUT, "face_lips").start, 0] += 0.2  # 0.2 / shoulder width 0.3
    report = compare(landmark_stats([base]), landmark_stats([moved]))
    assert report.xy_mean == pytest.approx(0.2 / 0.3, abs=0.01)
    assert report.xy_std < 0.01


def test_swapped_hands_show_up_as_a_large_shift() -> None:
    base = upright_person()
    swapped = base.copy()
    left, right = group_slice(LAYOUT, "hand_left"), group_slice(LAYOUT, "hand_right")
    swapped[:, left], swapped[:, right] = base[:, right], base[:, left]
    assert compare(landmark_stats([base]), landmark_stats([swapped])).xy_mean > 1.0


def test_presence_gap_is_reported() -> None:
    base = upright_person()
    sparse = base.copy()
    sparse[::2, group_slice(LAYOUT, "hand_left")] = np.nan
    report = compare(landmark_stats([base]), landmark_stats([sparse]))
    assert report.presence == pytest.approx(0.5)


def test_frames_without_shoulders_are_ignored() -> None:
    base = upright_person()
    broken = base.copy()
    broken[0, LEFT_SHOULDER] = np.nan  # voids frame 0 in normalization
    assert landmark_stats([broken]).frames == base.shape[0] - 1


def test_no_valid_frames_is_an_error() -> None:
    nothing = np.full((2, N, 3), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="no valid frames"):
        landmark_stats([nothing])
