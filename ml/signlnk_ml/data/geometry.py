"""Anatomical sanity checks on slk-landmarks-v1 tensors.

These catch left/right swaps and wrong face indices without needing matching video: in any real
upright, unmirrored recording the left wrist is nearer the left hand than the right hand, lips are
below the nose tip, brows are above the eyes, and each eye group sits next to its BlazePose eye.
The same function runs on Kaggle ISLR (mapped from Holistic) and on browser recordings (Tasks),
so one passing run on each side shows they index the same anatomy.

Coordinates are image space: y grows downward, and the subject's left is on the image's right.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from signlnk_ml.features.normalize import load_layout
from signlnk_ml.schemas_generated.landmark_layout_v1 import LandmarkLayoutV1

Points = npt.NDArray[np.float64]  # [T, 2]

# BlazePose indices used as anatomical references.
POSE_LEFT_EYE, POSE_RIGHT_EYE = 2, 5
POSE_LEFT_WRIST, POSE_RIGHT_WRIST = 15, 16


def group_slice(layout: LandmarkLayoutV1, name: str) -> slice:
    """Output-landmark range of a layout group."""
    start = 0
    for group in layout.groups:
        if group.name == name:
            return slice(start, start + len(group.indices))
        start += len(group.indices)
    raise KeyError(name)


def pose_position(layout: LandmarkLayoutV1, blazepose_index: int) -> int:
    """Output position of a BlazePose landmark in the layout's pose group."""
    pose = next(g for g in layout.groups if g.name == "pose")
    offset = [int(i.root) for i in pose.indices].index(blazepose_index)
    return int(group_slice(layout, "pose").start) + offset


def _distance(p: Points, q: Points) -> npt.NDArray[np.float64]:
    distances: npt.NDArray[np.float64] = np.linalg.norm(p - q, axis=1)
    return distances


def _fraction(condition: npt.NDArray[np.bool_], *points: Points) -> float:
    """Share of frames (where every point is finite) in which `condition` holds; NaN if none."""
    valid = np.logical_and.reduce([np.isfinite(p).all(axis=1) for p in points])
    return float(condition[valid].mean()) if valid.any() else float("nan")


def geometry_checks(
    sequence: npt.ArrayLike, layout: LandmarkLayoutV1 | None = None
) -> dict[str, float]:
    """Fraction of valid frames satisfying each anatomical expectation (NaN if no valid frame).

    `sequence` is raw (un-normalized) landmarks [T, N, 3].
    """
    layout = layout or load_layout()
    xy = np.asarray(sequence, dtype=np.float64)[..., :2]

    def pose(index: int) -> Points:
        return xy[:, pose_position(layout, index)]

    def first(group: str) -> Points:
        return xy[:, group_slice(layout, group).start]

    def centroid(group: str) -> Points:
        mean: Points = xy[:, group_slice(layout, group)].mean(axis=1)
        return mean

    left_wrist, right_wrist = pose(POSE_LEFT_WRIST), pose(POSE_RIGHT_WRIST)
    left_eye_ref, right_eye_ref = pose(POSE_LEFT_EYE), pose(POSE_RIGHT_EYE)
    left_hand, right_hand, nose = first("hand_left"), first("hand_right"), first("face_nose_tip")
    lips = centroid("face_lips")
    left_eye, right_eye = centroid("face_left_eye"), centroid("face_right_eye")
    left_brow, right_brow = centroid("face_left_eyebrow"), centroid("face_right_eyebrow")

    return {
        "left_hand_at_left_wrist": _fraction(
            _distance(left_hand, left_wrist) < _distance(left_hand, right_wrist),
            left_hand,
            left_wrist,
            right_wrist,
        ),
        "right_hand_at_right_wrist": _fraction(
            _distance(right_hand, right_wrist) < _distance(right_hand, left_wrist),
            right_hand,
            left_wrist,
            right_wrist,
        ),
        "lips_below_nose": _fraction(lips[:, 1] > nose[:, 1], lips, nose),
        "left_brow_above_left_eye": _fraction(
            left_brow[:, 1] < left_eye[:, 1], left_brow, left_eye
        ),
        "right_brow_above_right_eye": _fraction(
            right_brow[:, 1] < right_eye[:, 1], right_brow, right_eye
        ),
        "left_eye_at_pose_left_eye": _fraction(
            _distance(left_eye, left_eye_ref) < _distance(left_eye, right_eye_ref),
            left_eye,
            left_eye_ref,
            right_eye_ref,
        ),
        "right_eye_at_pose_right_eye": _fraction(
            _distance(right_eye, right_eye_ref) < _distance(right_eye, left_eye_ref),
            right_eye,
            left_eye_ref,
            right_eye_ref,
        ),
        "subject_left_is_image_right": _fraction(
            left_eye[:, 0] > right_eye[:, 0], left_eye, right_eye
        ),
    }
