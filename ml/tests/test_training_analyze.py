"""Offline analysis of saved scores: vocabulary subsets, ensembles, confusion matrix (step 4)."""

from pathlib import Path

import numpy as np
import pytest
from signlnk_ml.training.analyze import (
    confusion_matrix,
    ensemble,
    load_scores,
    random_subset_top1,
    save_scores,
    subset_top1,
)

SCORES = np.array(
    [
        [0.7, 0.2, 0.1, 0.0],  # label 0, predicted 0
        [0.1, 0.3, 0.6, 0.0],  # label 1, predicted 2 (3-way); inside {0, 1}: predicted 1
        [0.0, 0.2, 0.3, 0.5],  # label 3, predicted 3
        [0.1, 0.2, 0.3, 0.4],  # label 2, predicted 3
    ]
)
LABELS = np.array([0, 1, 3, 2])


def test_subset_top1_scores_only_rows_in_the_subset_against_each_other() -> None:
    # rows with labels 0 and 1; the competitor classes 2 and 3 are removed
    assert subset_top1(SCORES, LABELS, [0, 1]) == pytest.approx(1.0)
    # full model on the same rows would get row 1 wrong
    assert float((SCORES.argmax(axis=1) == LABELS)[[0, 1]].mean()) == pytest.approx(0.5)


def test_subset_top1_never_lowers_accuracy_for_rows_whose_label_is_kept() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random((200, 10))
    labels = rng.integers(0, 10, 200)
    subset = [1, 3, 5, 7]
    keep = np.isin(labels, subset)
    unrestricted = float((scores.argmax(axis=1) == labels)[keep].mean())
    assert subset_top1(scores, labels, subset) >= unrestricted


def test_subset_top1_rejects_an_empty_subset_or_no_matching_rows() -> None:
    with pytest.raises(ValueError, match="empty"):
        subset_top1(SCORES, LABELS, [])
    with pytest.raises(ValueError, match="no windows"):
        subset_top1(SCORES, np.array([0, 0, 0, 0]), [1, 2])


def test_random_subset_top1_is_seeded_and_summarises_the_draws() -> None:
    rng = np.random.default_rng(1)
    scores = rng.random((400, 12))
    labels = rng.integers(0, 12, 400)
    a = random_subset_top1(scores, labels, size=4, draws=30, seed=3, exclude=[11])
    b = random_subset_top1(scores, labels, size=4, draws=30, seed=3, exclude=[11])
    assert a == b
    assert a["min"] <= a["p10"] <= a["max"] and a["min"] <= a["mean"] <= a["max"]
    assert a["draws"] == 30 and a["size"] == 4


def test_smaller_vocabularies_are_easier_on_average() -> None:
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 20, 2000)
    scores = rng.random((2000, 20)) + np.eye(20)[labels] * 0.4  # an informative but noisy model
    small = random_subset_top1(scores, labels, size=3, draws=60, seed=0)
    large = random_subset_top1(scores, labels, size=15, draws=60, seed=0)
    assert small["mean"] > large["mean"]


def test_random_subset_rejects_a_size_larger_than_the_classes() -> None:
    with pytest.raises(ValueError, match="size"):
        random_subset_top1(SCORES, LABELS, size=5, draws=3, seed=0)
    with pytest.raises(ValueError, match="size"):
        random_subset_top1(SCORES, LABELS, size=4, draws=3, seed=0, exclude=[3])


def test_random_subsets_only_use_classes_that_have_windows() -> None:
    scores = np.random.default_rng(4).random((60, 12))
    labels = np.repeat([0, 1, 2], 20)  # classes 3..11 never occur
    result = random_subset_top1(scores, labels, size=2, draws=50, seed=1)
    assert 0.0 <= result["min"] <= result["max"] <= 1.0
    with pytest.raises(ValueError, match="size"):
        random_subset_top1(scores, labels, size=4, draws=5, seed=1)


def test_confusion_matrix_counts_true_by_predicted() -> None:
    matrix = confusion_matrix(SCORES, LABELS, n_classes=4)
    assert matrix.shape == (4, 4) and matrix.sum() == 4
    assert matrix[0, 0] == 1 and matrix[1, 2] == 1 and matrix[3, 3] == 1 and matrix[2, 3] == 1


def test_ensemble_averages_scores_and_requires_identical_rows() -> None:
    a, b = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
    assert np.allclose(ensemble([a, b]), [[0.5, 0.5]])
    with pytest.raises(ValueError, match="same shape"):
        ensemble([a, np.zeros((2, 2))])


def test_saved_scores_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "scores.npz"
    save_scores(
        path,
        {
            "val": (SCORES, LABELS, np.array(["a", "a", "b", "b"])),
            "test": (SCORES[:2], LABELS[:2], np.array(["c", "c"])),
        },
    )
    loaded = load_scores(path)
    assert set(loaded) == {"val", "test"}
    scores, labels, participants = loaded["val"]
    assert np.allclose(scores, SCORES, atol=1e-3) and labels.tolist() == LABELS.tolist()
    assert participants.tolist() == ["a", "a", "b", "b"]
    assert scores.dtype == np.float32
