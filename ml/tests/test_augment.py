"""Augmentations on normalized landmark windows [T, N, 3] (Phase 1 step 3)."""

import numpy as np
from signlnk_ml.data.geometry import group_slice
from signlnk_ml.features.normalize import load_layout
from signlnk_ml.training.augment import (
    Augmenter,
    frame_drop,
    mirror,
    rotate,
    scale,
    time_stretch,
)

LAYOUT = load_layout()
N = LAYOUT.n_landmarks
FACE = np.concatenate(
    [np.arange(N)[group_slice(LAYOUT, g.name)] for g in LAYOUT.groups if g.name.startswith("face")]
)
NOT_FACE = np.setdiff1d(np.arange(N), FACE)


def window(frames: int = 8, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(frames, N, 3)).astype(np.float32)


def test_mirror_swaps_hands_and_negates_x() -> None:
    w = window()
    m = mirror(w, LAYOUT)
    left, right = group_slice(LAYOUT, "hand_left"), group_slice(LAYOUT, "hand_right")
    assert np.allclose(m[:, left, 0], -w[:, right, 0])
    assert np.allclose(m[:, left, 1:], w[:, right, 1:])
    assert np.allclose(m[:, right, 0], -w[:, left, 0])


def test_mirror_swaps_pose_pairs_and_keeps_the_nose_on_the_midline() -> None:
    w = window()
    m = mirror(w, LAYOUT)
    pose = np.arange(N)[group_slice(LAYOUT, "pose")]
    # layout pose order: 0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16 (BlazePose indices)
    assert np.allclose(m[:, pose[0], 0], -w[:, pose[0], 0])  # nose
    for a, b in [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]:
        assert np.allclose(m[:, pose[a], 0], -w[:, pose[b], 0])
        assert np.allclose(m[:, pose[b], 1], w[:, pose[a], 1])


def test_mirror_twice_restores_everything_but_the_face() -> None:
    w = window()
    twice = mirror(mirror(w, LAYOUT), LAYOUT)
    assert np.allclose(twice[:, NOT_FACE], w[:, NOT_FACE])


def test_mirror_marks_the_face_missing() -> None:
    """Face landmarks have no verified left/right pairing, so mirrored faces are dropped."""
    assert np.isnan(mirror(window(), LAYOUT)[:, FACE]).all()


def test_mirror_does_not_modify_its_input() -> None:
    w = window()
    before = w.copy()
    mirror(w, LAYOUT)
    assert np.array_equal(w, before)


def test_rotate_preserves_xy_radius_and_leaves_z_alone() -> None:
    w = window()
    r = rotate(w, np.deg2rad(15))
    assert np.allclose(np.hypot(r[..., 0], r[..., 1]), np.hypot(w[..., 0], w[..., 1]), atol=1e-5)
    assert np.array_equal(r[..., 2], w[..., 2])
    assert not np.allclose(r[..., 0], w[..., 0])


def test_rotate_by_zero_is_identity_and_nan_stays_nan() -> None:
    w = window()
    w[1, 3] = np.nan
    r = rotate(w, 0.0)
    assert np.allclose(r, w, equal_nan=True)
    assert np.isnan(rotate(w, 0.3)[1, 3]).all()


def test_scale_multiplies_all_coordinates() -> None:
    w = window()
    assert np.allclose(scale(w, 1.1), w * 1.1)


def test_time_stretch_slower_repeats_frames_and_faster_pads_with_missing() -> None:
    w = np.arange(8, dtype=np.float32).reshape(8, 1, 1) * np.ones((8, N, 3), np.float32)
    slow = time_stretch(w, 2.0)
    assert slow.shape == w.shape and slow[:, 0, 0].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    fast = time_stretch(w, 0.5)
    assert fast[:4, 0, 0].tolist() == [0, 2, 4, 6] and np.isnan(fast[4:]).all()
    assert np.array_equal(time_stretch(w, 1.0), w)


def test_frame_drop_extremes_and_determinism() -> None:
    w = window(20)
    assert np.array_equal(frame_drop(w, 0.0, np.random.default_rng(1)), w)
    assert np.isnan(frame_drop(w, 1.0, np.random.default_rng(1))).all()
    a = frame_drop(w, 0.3, np.random.default_rng(5))
    b = frame_drop(w, 0.3, np.random.default_rng(5))
    assert np.array_equal(a, b, equal_nan=True)
    dropped = np.isnan(a).all(axis=(1, 2))
    assert 0 < dropped.sum() < 20 and np.array_equal(a[~dropped], w[~dropped])


def test_augmenter_is_seeded_and_keeps_shape_and_dtype() -> None:
    w = window(16)
    one = Augmenter(LAYOUT, seed=3)(w)
    two = Augmenter(LAYOUT, seed=3)(w)
    assert one.shape == w.shape and one.dtype == np.float32
    assert np.array_equal(one, two, equal_nan=True)


def test_augmenter_changes_the_window() -> None:
    w = window(16)
    changed = sum(
        not np.array_equal(Augmenter(LAYOUT, seed=s)(w), w, equal_nan=True) for s in range(20)
    )
    assert changed == 20
