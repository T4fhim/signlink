"""Top-k accuracy and the per-signer report (Phase 1 step 3)."""

import numpy as np
import pytest
from signlnk_ml.training.evaluate import per_participant_top1, report, topk_accuracy

SCORES = np.array(
    [
        [0.7, 0.2, 0.05, 0.05],  # label 0: top-1 hit
        [0.1, 0.5, 0.3, 0.1],  # label 2: top-2 hit only
        [0.4, 0.3, 0.2, 0.1],  # label 3: predicted 0, a hit only at k=4
        [0.05, 0.05, 0.1, 0.8],  # label 3: top-1 hit
    ]
)
LABELS = np.array([0, 2, 3, 3])


def test_topk_accuracy_counts_hits_within_k() -> None:
    assert topk_accuracy(SCORES, LABELS, 1) == pytest.approx(0.5)
    assert topk_accuracy(SCORES, LABELS, 2) == pytest.approx(0.75)
    assert topk_accuracy(SCORES, LABELS, 4) == 1.0


def test_topk_rejects_bad_k_and_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="k must be"):
        topk_accuracy(SCORES, LABELS, 0)
    with pytest.raises(ValueError, match="same length"):
        topk_accuracy(SCORES, LABELS[:2], 1)


def test_per_participant_top1_groups_by_signer() -> None:
    participants = np.array(["a", "a", "b", "b"])
    assert per_participant_top1(SCORES, LABELS, participants) == {"a": 0.5, "b": 0.5}


def test_report_has_overall_numbers_signers_and_the_other_class() -> None:
    participants = np.array(["a", "a", "b", "b"])
    result = report(SCORES, LABELS, participants, other_index=3)
    assert result["n"] == 4
    assert result["top1"] == pytest.approx(0.5)
    assert result["top5"] == 1.0  # k is capped at the number of classes
    assert result["per_participant_top1"] == {"a": 0.5, "b": 0.5}
    assert result["min_participant_top1"] == 0.5
    # two 'other' windows (label 3); one is predicted as sign 0, one correctly as 'other'
    assert result["n_other"] == 2 and result["false_activation_rate"] == pytest.approx(0.5)


def test_false_activation_rate_counts_other_windows_predicted_as_a_sign() -> None:
    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    labels = np.array([1, 1, 0])  # class 1 is 'other'
    result = report(scores, labels, np.array(["a", "a", "b"]), other_index=1)
    assert result["n_other"] == 2
    assert result["false_activation_rate"] == pytest.approx(0.5)
    assert result["top1_signs_only"] == 1.0
