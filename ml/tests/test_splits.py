"""Signer-independent splits (Phase 1 step 2): no participant in more than one split."""

import csv
from pathlib import Path

import pytest
from signlnk_ml.data.splits import (
    SPLIT_NAMES,
    SPLITS_PATH,
    assert_handedness_balanced,
    assert_no_overlap,
    assign_participants,
    dominant_hand,
    load_assignment,
    load_handedness,
)

PARTICIPANTS = [str(1000 + 7 * i) for i in range(21)]


def rows_for(participants: list[str]) -> list[dict[str, str]]:
    return [{"participant_id": p, "sign": "a"} for p in participants for _ in range(2)]


def test_assignment_is_deterministic_and_order_independent() -> None:
    first = assign_participants(PARTICIPANTS, val=3, test=3, seed=7)
    again = assign_participants(list(reversed(PARTICIPANTS)), val=3, test=3, seed=7)
    assert first == again


def test_assignment_sizes_and_coverage() -> None:
    assignment = assign_participants(PARTICIPANTS, val=3, test=3, seed=7)
    counts = {name: list(assignment.values()).count(name) for name in SPLIT_NAMES}
    assert counts == {"train": 15, "val": 3, "test": 3}
    assert set(assignment) == set(PARTICIPANTS)


def test_a_different_seed_gives_a_different_split() -> None:
    a = assign_participants(PARTICIPANTS, val=3, test=3, seed=1)
    b = assign_participants(PARTICIPANTS, val=3, test=3, seed=2)
    assert a != b


def test_too_few_participants_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least"):
        assign_participants(["1", "2"], val=1, test=1, seed=0)


HANDS = {
    p: ("left" if i % 3 == 0 else "mixed" if i % 7 == 0 else "right")
    for i, p in enumerate(PARTICIPANTS)
}


def test_handedness_stratified_split_has_both_hands_in_val_and_test() -> None:
    assignment = assign_participants(PARTICIPANTS, val=3, test=3, seed=7, hands=HANDS)
    assert_handedness_balanced(assignment, HANDS)
    for name in ("val", "test"):
        labels = {HANDS[p] for p, s in assignment.items() if s == name}
        assert {"left", "right"} <= labels


def test_unbalanced_handedness_is_caught() -> None:
    assignment = dict.fromkeys(PARTICIPANTS, "train")
    right = [p for p in PARTICIPANTS if HANDS[p] == "right"]
    for p in right[:3]:
        assignment[p] = "val"
    for p in right[3:6]:
        assignment[p] = "test"
    with pytest.raises(AssertionError, match="left-dominant"):
        assert_handedness_balanced(assignment, HANDS)


def test_impossible_handedness_constraint_is_rejected() -> None:
    all_right = dict.fromkeys(PARTICIPANTS, "right")
    with pytest.raises(ValueError, match="no seed"):
        assign_participants(PARTICIPANTS, val=3, test=3, seed=7, hands=all_right)


def test_dominant_hand_labels() -> None:
    assert dominant_hand(0.6, 0.0) == "left"
    assert dominant_hand(0.0, 0.5) == "right"
    assert dominant_hand(0.32, 0.28) == "mixed"
    assert dominant_hand(0.0, 0.0) == "mixed"


def test_duplicate_participant_in_the_file_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "split.json"
    bad.write_text(
        '{"dataset": "kaggle-islr", "participants": {"1": "train", "1": "test"}}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_assignment(bad)


def test_clean_assignment_passes() -> None:
    assignment = assign_participants(PARTICIPANTS, val=3, test=3, seed=7)
    assert_no_overlap(assignment, rows_for(PARTICIPANTS))


def test_unassigned_participant_is_caught() -> None:
    assignment = assign_participants(PARTICIPANTS, val=3, test=3, seed=7)
    del assignment[PARTICIPANTS[0]]
    with pytest.raises(AssertionError, match="not assigned"):
        assert_no_overlap(assignment, rows_for(PARTICIPANTS))


def test_empty_split_is_caught() -> None:
    assignment = dict.fromkeys(PARTICIPANTS, "train")
    with pytest.raises(AssertionError, match="empty"):
        assert_no_overlap(assignment, rows_for(PARTICIPANTS))


def test_unknown_split_name_is_caught() -> None:
    assignment = assign_participants(PARTICIPANTS, val=3, test=3, seed=7)
    assignment[PARTICIPANTS[0]] = "dev"
    with pytest.raises(AssertionError, match="unknown split"):
        assert_no_overlap(assignment, rows_for(PARTICIPANTS))


def test_a_participant_in_two_splits_is_caught() -> None:
    """The overlap check works on rows, so a sequence tagged with a second split is caught."""
    assignment = assign_participants(PARTICIPANTS, val=3, test=3, seed=7)
    rows = rows_for(PARTICIPANTS)
    rows.append({"participant_id": PARTICIPANTS[0], "sign": "a", "split": "other"})
    with pytest.raises(AssertionError, match="more than one split"):
        assert_no_overlap(assignment, rows)


def test_committed_assignment_is_clean() -> None:
    """Runs in CI without data: three non-empty, pairwise-disjoint participant sets."""
    assignment = load_assignment(SPLITS_PATH)
    sets = {name: {p for p, s in assignment.items() if s == name} for name in SPLIT_NAMES}
    assert all(sets.values())
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])


def test_committed_split_has_both_hands_in_val_and_test() -> None:
    assert_handedness_balanced(load_assignment(SPLITS_PATH), load_handedness(SPLITS_PATH))


def test_committed_assignment_covers_kaggle_train_csv(islr_root: Path) -> None:
    with (islr_root / "train.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert_no_overlap(load_assignment(SPLITS_PATH), rows)
    assert {r["participant_id"] for r in rows} == set(load_assignment(SPLITS_PATH))
