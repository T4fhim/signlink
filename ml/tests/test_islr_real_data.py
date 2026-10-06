"""Train/serve parity on real data: the Kaggle ISLR sample and, when present, browser recordings.

Skipped when the data is not on this machine. Run locally with
`SIGNLNK_DATA_DIR=D:\\signlnk-data uv run pytest ml/tests/test_islr_real_data.py`.

The limits below come from a leave-one-participant-out baseline on the 525-sequence, 21-participant
sample (ADR-0006): each limit is about 1.5 to 2 times the worst of the 21 held-out participants, so
natural between-person variation passes and a systematic pipeline difference does not.
"""

import hashlib
import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
from signlnk_ml.data.geometry import geometry_checks, group_slice
from signlnk_ml.data.parity import ParityReport, compare, landmark_stats, parity_groups
from signlnk_ml.data.recordings import load_recording, recordings_dir
from signlnk_ml.features.normalize import load_layout

Array = npt.NDArray[np.float32]
Sequences = dict[str, list[Array]]
LAYOUT = load_layout()


@dataclass(frozen=True)
class Limits:
    """Largest allowed worst-landmark difference (normalized units; presence is a fraction)."""

    xy_mean: float
    xy_std: float
    z_mean: float = float("inf")
    z_std: float = float("inf")
    presence: float = 1.0


# Baseline worst-of-21: face xy_mean 0.128, xy_std 0.051, z_mean 0.046, z_std 0.015, presence 0.109.
FACE_LIMITS = Limits(xy_mean=0.20, xy_std=0.10, z_mean=0.10, z_std=0.04, presence=0.25)
# Baseline worst-of-21: head xy_mean 0.116, xy_std 0.048, presence 0. Pose z depends on the camera
# (worst z_mean 0.72), so it is not compared.
HEAD_LIMITS = Limits(xy_mean=0.20, xy_std=0.10, presence=0.05)
# Observed on the sample: every check averages >= 0.986 (Holistic mislabels a hand in a few cases).
MIN_GEOMETRY_AGREEMENT = 0.97
# Browser recordings are short and few, so geometry is pooled over all usable clips and held to a
# looser bar than Kaggle. Observed on 6 usable clips (540 frames, label-based hand assignment): left
# hand 0.928, right hand 0.994, every face and orientation check 1.000.
MIN_RECORDING_GEOMETRY = 0.90
# A clip needs a face and a person in most frames to say anything about face/head parity.
MIN_PART_PRESENCE = 0.8
MIN_USABLE_RECORDINGS = 2


def violations(report: ParityReport, limits: Limits) -> list[str]:
    found = []
    for name in ("xy_mean", "xy_std", "z_mean", "z_std", "presence"):
        value, limit = getattr(report, name), getattr(limits, name)
        if not value <= limit:  # also catches NaN
            found.append(f"{name}={value:.3f} > {limit}")
    return found


def everything(sample: Sequences) -> list[Array]:
    return [s for sequences in sample.values() for s in sequences]


def test_sample_is_loadable_and_complete(islr_sample: Sequences) -> None:
    assert len(islr_sample) == 21
    sequences = everything(islr_sample)
    assert len(sequences) >= 500
    for s in sequences:
        assert s.dtype == np.float32
        assert s.shape[1:] == (LAYOUT.n_landmarks, 3)
        assert s.shape[0] >= 1
    pose = group_slice(LAYOUT, "pose")
    frames = np.concatenate(sequences)
    assert np.isfinite(frames[:, pose, 0]).all(axis=1).mean() > 0.9  # a person is in view


def test_geometry_holds_on_kaggle_data(islr_sample: Sequences) -> None:
    per_check: dict[str, list[float]] = {}
    for s in everything(islr_sample):
        for name, value in geometry_checks(s).items():
            if np.isfinite(value):
                per_check.setdefault(name, []).append(value)
    assert len(per_check) == 8
    for name, values in per_check.items():
        assert np.mean(values) >= MIN_GEOMETRY_AGREEMENT, f"{name}: {np.mean(values):.3f}"


def test_leave_one_participant_out_stays_within_limits(islr_sample: Sequences) -> None:
    groups = parity_groups(LAYOUT)
    problems = []
    for held_out, own in islr_sample.items():
        rest = [s for p, seqs in islr_sample.items() if p != held_out for s in seqs]
        train, serve = landmark_stats(rest), landmark_stats(own)
        for name, limits in (("face", FACE_LIMITS), ("head", HEAD_LIMITS)):
            for issue in violations(compare(train, serve, groups[name]), limits):
                problems.append(f"participant {held_out} {name}: {issue}")
    assert not problems, problems


def test_a_wrong_face_mapping_is_caught(islr_sample: Sequences) -> None:
    """Sensitivity: scrambling which face index is which must exceed the face limit."""
    face = parity_groups(LAYOUT)["face"]
    held_out = sorted(islr_sample)[0]
    rest = landmark_stats([s for p, seqs in islr_sample.items() if p != held_out for s in seqs])
    scrambled = []
    for s in islr_sample[held_out]:
        t = s.copy()
        t[:, face] = s[:, np.roll(face, 40)]  # lips take the eyebrows' places, and so on
        scrambled.append(t)
    report = compare(rest, landmark_stats(scrambled), face)
    assert report.xy_mean > FACE_LIMITS.xy_mean


def test_swapped_hands_are_caught_by_geometry(islr_sample: Sequences) -> None:
    left, right = group_slice(LAYOUT, "hand_left"), group_slice(LAYOUT, "hand_right")
    held_out = sorted(islr_sample)[0]
    agreement = {"as_loaded": [], "swapped": []}  # type: dict[str, list[float]]
    for s in islr_sample[held_out]:
        swapped = s.copy()
        swapped[:, left], swapped[:, right] = s[:, right], s[:, left]
        for label, sequence in (("as_loaded", s), ("swapped", swapped)):
            checks = geometry_checks(sequence)
            agreement[label] += [
                v
                for k, v in checks.items()
                if k in ("left_hand_at_left_wrist", "right_hand_at_right_wrist") and np.isfinite(v)
            ]
    assert np.mean(agreement["as_loaded"]) > 0.9
    assert np.mean(agreement["swapped"]) < 0.2


def usable_recordings() -> list[tuple[str, Array]]:
    """Recordings with a face and pose in most frames; duplicates and bad clips are set aside."""
    usable: list[tuple[str, Array]] = []
    seen: set[str] = set()
    for path in sorted(recordings_dir().glob("*.npy")):
        recording = load_recording(path)
        digest = hashlib.sha256(recording.tobytes()).hexdigest()
        if digest in seen:
            warnings.warn(f"{path.name}: identical to an earlier recording, ignored", stacklevel=2)
            continue
        seen.add(digest)
        face = float(
            np.isfinite(recording[:, group_slice(LAYOUT, "face_lips"), 0]).all(axis=1).mean()
        )
        pose = float(np.isfinite(recording[:, group_slice(LAYOUT, "pose"), 0]).all(axis=1).mean())
        if min(face, pose) < MIN_PART_PRESENCE:
            warnings.warn(
                f"{path.name}: face found in {face:.0%} and pose in {pose:.0%} of frames, "
                "ignored: re-record with the face and shoulders in view",
                stacklevel=2,
            )
            continue
        usable.append((path.name, recording))
    return usable


def test_browser_recordings_match_the_training_data(islr_sample: Sequences) -> None:
    if not list(recordings_dir().glob("*.npy")):
        pytest.skip("no browser recordings yet: record some with /dev/record (Phase 0 step 6)")
    clips = usable_recordings()
    assert len(clips) >= MIN_USABLE_RECORDINGS, (
        f"only {len(clips)} usable recording(s); need {MIN_USABLE_RECORDINGS}"
    )

    pooled = geometry_checks(np.concatenate([recording for _, recording in clips]))
    assert np.isfinite(pooled["lips_below_nose"]), "no face found in any recording"
    for name, value in pooled.items():
        assert np.isnan(value) or value >= MIN_RECORDING_GEOMETRY, f"pooled {name}: {value:.3f}"

    train = landmark_stats(everything(islr_sample))
    groups = parity_groups(LAYOUT)
    problems = []
    for name, recording in clips:
        serve = landmark_stats([recording])
        found = violations(compare(train, serve, groups["face"]), FACE_LIMITS)
        found += violations(compare(train, serve, groups["head"]), HEAD_LIMITS)
        problems += [f"{name}: {issue}" for issue in found]
    assert not problems, problems
