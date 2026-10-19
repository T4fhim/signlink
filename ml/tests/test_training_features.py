"""Fixed-length windows, hands-absent runs and model inputs (Phase 1 step 3)."""

import numpy as np
import pytest
from signlnk_ml.data.geometry import group_slice
from signlnk_ml.features.normalize import load_layout
from signlnk_ml.training.features import (
    hands_absent_runs,
    input_dim,
    make_input,
    resample_nearest,
)

LAYOUT = load_layout()
N = LAYOUT.n_landmarks


def sequence(frames: int, hands_present: list[bool]) -> np.ndarray:
    seq = np.random.default_rng(0).normal(size=(frames, N, 3)).astype(np.float32)
    for frame, present in enumerate(hands_present):
        if not present:
            seq[frame, group_slice(LAYOUT, "hand_left")] = np.nan
            seq[frame, group_slice(LAYOUT, "hand_right")] = np.nan
    return seq


def test_resample_upsamples_by_repeating_nearest_frames() -> None:
    seq = np.arange(3, dtype=np.float32).reshape(3, 1, 1) * np.ones((3, N, 3), np.float32)
    out = resample_nearest(seq, 6)
    assert out.shape == (6, N, 3)
    assert out[:, 0, 0].tolist() == [0, 0, 1, 1, 2, 2]


def test_resample_downsamples_and_keeps_length_one() -> None:
    seq = np.arange(8, dtype=np.float32).reshape(8, 1, 1) * np.ones((8, N, 3), np.float32)
    assert resample_nearest(seq, 4)[:, 0, 0].tolist() == [0, 2, 4, 6]
    assert resample_nearest(seq[:1], 5).shape == (5, N, 3)


def test_resample_never_interpolates_missing_values() -> None:
    seq = np.ones((3, N, 3), np.float32)
    seq[1] = np.nan
    out = resample_nearest(seq, 9)
    assert set(np.isnan(out[:, 0, 0]).tolist()) == {True, False}
    assert np.isin(out[~np.isnan(out)], [1.0]).all()


def test_resample_rejects_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="empty"):
        resample_nearest(np.zeros((0, N, 3), np.float32), 4)


def test_hands_absent_runs_finds_maximal_runs_of_at_least_min_run() -> None:
    present = [True] * 3 + [False] * 10 + [True] * 2 + [False] * 4 + [True]
    seq = sequence(len(present), present)
    assert hands_absent_runs(seq, LAYOUT, min_run=8) == [(3, 13)]
    assert hands_absent_runs(seq, LAYOUT, min_run=4) == [(3, 13), (15, 19)]


def test_one_visible_hand_is_not_absent() -> None:
    seq = sequence(12, [False] * 12)
    seq[:, group_slice(LAYOUT, "hand_right")] = 0.5
    assert hands_absent_runs(seq, LAYOUT, min_run=2) == []


def test_make_input_has_no_nan_and_a_mask_per_landmark() -> None:
    window = sequence(5, [True] * 5)
    window[2, 10] = np.nan
    window[:, group_slice(LAYOUT, "face_lips")] = np.nan
    x = make_input(window, use_z=True)
    assert x.shape == (5, N * 3 + N) and x.dtype == np.float32
    assert not np.isnan(x).any()
    mask = x[:, N * 3 :]
    assert mask[2, 10] == 0 and mask[2, 11] == 1
    assert (mask[:, group_slice(LAYOUT, "face_lips")] == 0).all()
    assert (x[2, 30:33] == 0).all()  # a missing landmark's coordinates are zero


def test_velocity_adds_frame_differences_where_both_frames_are_present() -> None:
    window = sequence(4, [True] * 4)
    window[2, 5] = np.nan  # landmark 5 missing in frame 2
    x = make_input(window, use_z=False, velocity=True)
    assert x.shape == (4, input_dim(N, use_z=False, velocity=True))
    velocity = x[:, N * 3 :].reshape(4, N, 2)
    assert np.allclose(velocity[0], 0)  # no previous frame
    assert np.allclose(velocity[1, 0], window[1, 0, :2] - window[0, 0, :2])
    assert np.allclose(velocity[2, 5], 0) and np.allclose(velocity[3, 5], 0)  # gap on either side
    assert not np.isnan(x).any()


def test_input_dim_matches_make_input_for_every_option() -> None:
    window = sequence(3, [True] * 3)
    for use_z in (False, True):
        for velocity in (False, True):
            x = make_input(window, use_z=use_z, velocity=velocity)
            assert x.shape[1] == input_dim(N, use_z=use_z, velocity=velocity)


def test_velocity_is_off_by_default() -> None:
    window = sequence(3, [True] * 3)
    assert make_input(window, use_z=False).shape[1] == N * 3


def test_make_input_without_z_drops_the_depth_channel() -> None:
    window = sequence(4, [True] * 4)
    x = make_input(window, use_z=False)
    assert x.shape == (4, N * 2 + N)
    assert np.array_equal(x[:, 0:2], window[:, 0, :2])
