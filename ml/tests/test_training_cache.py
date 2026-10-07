"""Windows from Kaggle sequences, and the 'other' (no sign) class (Phase 1 step 3)."""

import numpy as np
import pytest
from signlnk_ml.data.geometry import group_slice, pose_position
from signlnk_ml.data.kaggle_islr import LEGACY_OFFSETS, LEGACY_TOTAL, to_slk_v1
from signlnk_ml.features.normalize import load_layout
from signlnk_ml.training.cache import (
    cache_mismatches,
    cap_other,
    other_windows,
    select_indices,
    sequence_to_window,
)
from signlnk_ml.training.config import CONFIGS_DIR, load_config

LAYOUT = load_layout()
N = LAYOUT.n_landmarks


def legacy_sequence(frames: int, hands_present: list[bool] | None = None) -> np.ndarray:
    rng = np.random.default_rng(1)
    seq = rng.uniform(0.2, 0.8, size=(frames, LEGACY_TOTAL, 3)).astype(np.float32)
    pose = LEGACY_OFFSETS["pose"]
    seq[:, pose + 11] = (0.6, 0.5, 0.0)  # left shoulder
    seq[:, pose + 12] = (0.4, 0.5, 0.0)  # right shoulder
    for frame, present in enumerate(hands_present or [True] * frames):
        if not present:
            seq[frame, LEGACY_OFFSETS["left_hand"] : LEGACY_OFFSETS["left_hand"] + 21] = np.nan
            seq[frame, LEGACY_OFFSETS["right_hand"] : LEGACY_OFFSETS["right_hand"] + 21] = np.nan
    return seq


def test_sequence_to_window_is_normalized_and_fixed_length() -> None:
    window = sequence_to_window(legacy_sequence(21), LAYOUT, length=64)
    assert window.shape == (64, N, 3) and window.dtype == np.float32
    left, right = pose_position(LAYOUT, 11), pose_position(LAYOUT, 12)
    assert np.allclose(window[:, left, 0], 0.5, atol=1e-5)  # shoulder width scaled to 1
    assert np.allclose(window[:, right, 0], -0.5, atol=1e-5)


def test_other_windows_come_only_from_hands_absent_runs() -> None:
    present = [True] * 5 + [False] * 12 + [True] * 4
    raw = to_slk_v1(legacy_sequence(len(present), present), LAYOUT)
    windows = other_windows(raw, LAYOUT, min_run=8, length=16, max_windows=3)
    assert len(windows) == 1 and windows[0].shape == (16, N, 3)
    for hand in ("hand_left", "hand_right"):
        assert np.isnan(windows[0][:, group_slice(LAYOUT, hand)]).all()
    assert np.isfinite(windows[0][:, group_slice(LAYOUT, "pose")]).all()  # still normalized


def test_other_windows_prefers_the_longest_runs_and_respects_the_cap() -> None:
    present = [False] * 9 + [True] * 2 + [False] * 15 + [True] * 2 + [False] * 11
    raw = to_slk_v1(legacy_sequence(len(present), present), LAYOUT)
    assert len(other_windows(raw, LAYOUT, min_run=8, length=8, max_windows=5)) == 3
    assert len(other_windows(raw, LAYOUT, min_run=8, length=8, max_windows=1)) == 1


def test_no_other_windows_when_hands_are_always_visible() -> None:
    raw = to_slk_v1(legacy_sequence(30), LAYOUT)
    assert other_windows(raw, LAYOUT, min_run=8, length=16, max_windows=3) == []


def test_cap_other_is_a_multiple_of_the_average_class_size() -> None:
    assert cap_other(available=10_000, n_sign_windows=25_000, n_sign_classes=250, ratio=2.0) == 200
    assert cap_other(available=50, n_sign_windows=25_000, n_sign_classes=250, ratio=2.0) == 50
    assert cap_other(available=100, n_sign_windows=0, n_sign_classes=250, ratio=2.0) == 0


def test_select_indices_is_a_sorted_deterministic_subset() -> None:
    a = select_indices(100, 10, seed=4)
    assert a.tolist() == select_indices(100, 10, seed=4).tolist()
    assert len(set(a.tolist())) == 10 and a.tolist() == sorted(a.tolist())
    assert select_indices(5, 10, seed=4).tolist() == [0, 1, 2, 3, 4]
    assert a.tolist() != select_indices(100, 10, seed=5).tolist()


def test_cache_mismatches_reports_settings_the_cache_was_not_built_with() -> None:
    cfg = load_config(CONFIGS_DIR / "baseline.yaml")
    meta = {"length": 64, "other_min_run": 8, "other_ratio": 2.0}
    assert cache_mismatches(meta, cfg) == []
    problems = cache_mismatches({**meta, "length": 32}, cfg)
    assert len(problems) == 1 and "length" in problems[0]
    assert cache_mismatches({"length": 64}, cfg)  # an old cache without the settings is rebuilt


def test_resample_length_must_be_positive() -> None:
    with pytest.raises(ValueError, match="length"):
        sequence_to_window(legacy_sequence(5), LAYOUT, length=0)
