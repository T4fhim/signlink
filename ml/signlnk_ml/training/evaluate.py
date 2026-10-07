"""Top-k accuracy, per-signer breakdown and false activations (PLAN §8). numpy only."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

Scores = npt.NDArray[np.floating[Any]]
Labels = npt.NDArray[np.integer[Any]]


def topk_accuracy(scores: Scores, labels: Labels, k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    k = min(k, scores.shape[1])
    top = np.argsort(-scores, axis=1)[:, :k]
    return float((top == labels[:, None]).any(axis=1).mean())


def per_participant_top1(
    scores: Scores, labels: Labels, participants: npt.NDArray[np.str_]
) -> dict[str, float]:
    hit = scores.argmax(axis=1) == labels
    return {
        str(p): float(hit[participants == p].mean()) for p in sorted(set(participants.tolist()))
    }


def report(
    scores: Scores,
    labels: Labels,
    participants: npt.NDArray[np.str_],
    other_index: int | None = None,
) -> dict[str, Any]:
    """Overall top-1/top-5, per-signer top-1, and (with an 'other' class) false activations."""
    per_signer = per_participant_top1(scores, labels, participants)
    result: dict[str, Any] = {
        "n": len(labels),
        "top1": topk_accuracy(scores, labels, 1),
        "top5": topk_accuracy(scores, labels, 5),
        "per_participant_top1": per_signer,
        "min_participant_top1": min(per_signer.values()),
        "n_other": 0,
        "false_activation_rate": None,
        "top1_signs_only": None,
    }
    if other_index is not None:
        predicted = scores.argmax(axis=1)
        is_other = labels == other_index
        result["n_other"] = int(is_other.sum())
        if is_other.any():
            result["false_activation_rate"] = float((predicted[is_other] != other_index).mean())
        if (~is_other).any():
            result["top1_signs_only"] = float((predicted[~is_other] == labels[~is_other]).mean())
    return result
