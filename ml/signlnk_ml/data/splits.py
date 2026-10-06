"""Signer-independent train/val/test splits for Kaggle ISLR (Phase 1 step 2).

Splits are by participant: every participant is in exactly one split, so a model is never
evaluated on a signer it trained on. The assignment is made once, deterministically from the sorted
participant IDs and a seed, and committed in `ml/splits/kaggle-islr-v1.json` so it never changes
silently. Participant IDs are Kaggle's anonymous numbers, not personal data.

Run (needs `train.csv`, see `fetch_islr`):
    uv run python -m signlnk_ml.data.splits --check   # assert the committed split is clean
    uv run python -m signlnk_ml.data.splits --write   # (re)create it; only for a new split version
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from signlnk_ml.data.fetch_islr import data_dir
from signlnk_ml.data.kaggle_islr import LEGACY_COUNTS, LEGACY_OFFSETS, read_sequence
from signlnk_ml.features.normalize import REPO_ROOT

SPLIT_NAMES = ("train", "val", "test")
SPLITS_PATH = REPO_ROOT / "ml" / "splits" / "kaggle-islr-v1.json"
DATASET = "kaggle-islr"
DEFAULT_SEED = 20261006
DEFAULT_VAL = 3
DEFAULT_TEST = 3


def _sorted_ids(participants: Iterable[str]) -> list[str]:
    return sorted(set(participants), key=lambda p: (len(p), p))


def dominant_hand(left_fraction: float, right_fraction: float) -> str:
    """'left' / 'right' when one hand is present in more than twice as many frames, else 'mixed'.

    Hands are Kaggle's `left_hand` / `right_hand` slots (Holistic attaches each to a pose wrist).
    """
    if left_fraction > 2 * right_fraction:
        return "left"
    if right_fraction > 2 * left_fraction:
        return "right"
    return "mixed"


def _held_out(ordered: list[str], val: int, test: int, seed: int) -> dict[str, str]:
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    held_out = {p: "test" for p in shuffled[:test]}
    held_out |= {p: "val" for p in shuffled[test : test + val]}
    return held_out


def _both_hands(assignment: Mapping[str, str], hands: Mapping[str, str], split: str) -> bool:
    labels = {hands[p] for p, s in assignment.items() if s == split}
    return {"left", "right"} <= labels


def assign_participants(
    participants: Iterable[str],
    val: int,
    test: int,
    seed: int,
    hands: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """participant_id -> split. Deterministic and independent of the input order.

    With `hands` (participant -> left|right|mixed), the first seed from `seed` upwards that puts at
    least one left- and one right-dominant participant in both val and test is used, so neither
    held-out split is measured on one hand only.
    """
    ordered = _sorted_ids(participants)
    if len(ordered) < val + test + 1:
        raise ValueError(f"need at least {val + test + 1} participants, got {len(ordered)}")
    for attempt in range(seed, seed + 1000 if hands else seed + 1):
        held_out = _held_out(ordered, val, test, attempt)
        assignment = {p: held_out.get(p, "train") for p in ordered}
        if hands is None or all(_both_hands(assignment, hands, s) for s in ("val", "test")):
            return assignment
    raise ValueError("no seed in 1000 tries gives val and test both hands")


def assert_no_overlap(assignment: Mapping[str, str], rows: Iterable[Mapping[str, str]]) -> None:
    """Raises AssertionError unless every row's participant is in exactly one, known, split."""
    unknown = sorted(set(assignment.values()) - set(SPLIT_NAMES))
    assert not unknown, f"unknown split name(s): {unknown}"
    seen: dict[str, set[str]] = {}
    for row in rows:
        participant = row["participant_id"]
        assert participant in assignment, f"participant {participant} is not assigned to a split"
        seen.setdefault(participant, set()).add(assignment[participant])
        if "split" in row:
            seen[participant].add(row["split"])
    crossing = sorted(p for p, splits in seen.items() if len(splits) > 1)
    assert not crossing, f"participant(s) in more than one split: {crossing}"
    for name in SPLIT_NAMES:
        assert any(assignment[p] == name for p in seen), f"split {name} is empty"


def assert_handedness_balanced(assignment: Mapping[str, str], hands: Mapping[str, str]) -> None:
    """Val and test each hold at least one left- and one right-dominant participant."""
    for split in ("val", "test"):
        for hand in ("left", "right"):
            assert any(hands[p] == hand for p, s in assignment.items() if s == split), (
                f"{split} has no {hand}-dominant participant"
            )


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate key(s) in split file: {duplicates}")
    return dict(pairs)


def _load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates
    )
    assert data["dataset"] == DATASET, f"{path} is not a {DATASET} split"
    return data


def load_assignment(path: Path = SPLITS_PATH) -> dict[str, str]:
    participants: dict[str, str] = _load(path)["participants"]
    return participants


def load_handedness(path: Path = SPLITS_PATH) -> dict[str, str]:
    hands: dict[str, str] = _load(path)["handedness"]
    return hands


HANDEDNESS_RULE = (
    "per participant, over the sample_manifest sequences: fraction of frames with a left_hand / "
    "right_hand slot present; dominant if more than 2x the other, else mixed"
)


def measure_handedness(root: Path) -> dict[str, str]:
    """participant -> left|right|mixed, measured on the sequences listed in sample_manifest.csv."""
    frames: dict[str, list[float]] = {}
    with (root / "sample_manifest.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            legacy = read_sequence(root / row["path"])
            counts = frames.setdefault(row["participant_id"], [0.0, 0.0, 0.0])
            counts[0] += len(legacy)
            for slot, name in ((1, "left_hand"), (2, "right_hand")):
                start = LEGACY_OFFSETS[name]
                counts[slot] += float(
                    np.isfinite(legacy[:, start : start + LEGACY_COUNTS[name], 0]).any(axis=1).sum()
                )
    return {p: dominant_hand(c[1] / c[0], c[2] / c[0]) for p, c in sorted(frames.items())}


def write_assignment(
    assignment: Mapping[str, str],
    seed: int,
    hands: Mapping[str, str],
    path: Path = SPLITS_PATH,
) -> None:
    payload = {
        "dataset": DATASET,
        "seed": seed,
        "handedness_rule": HANDEDNESS_RULE,
        "handedness": dict(hands),
        "participants": dict(assignment),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:  # LF on Windows too
        handle.write(json.dumps(payload, indent=2) + "\n")


def read_rows(root: Path) -> list[dict[str, str]]:
    with (root / "train.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(assignment: Mapping[str, str], rows: list[dict[str, str]]) -> str:
    all_signs = {r["sign"] for r in rows}
    lines = []
    for name in SPLIT_NAMES:
        mine = [r for r in rows if assignment[r["participant_id"]] == name]
        people = sum(1 for split in assignment.values() if split == name)
        signs = len({r["sign"] for r in mine})
        lines.append(
            f"{name}: {people} participants, {len(mine)} sequences, {signs}/{len(all_signs)} signs"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="assert the committed split is clean")
    mode.add_argument("--write", action="store_true", help="create the split file")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    rows = read_rows(data_dir())
    if args.write:
        participants = {r["participant_id"] for r in rows}
        hands = measure_handedness(data_dir())
        assignment = assign_participants(
            participants, DEFAULT_VAL, DEFAULT_TEST, args.seed, hands=hands
        )
        write_assignment(assignment, args.seed, hands)
    assignment = load_assignment()
    assert_no_overlap(assignment, rows)
    assert {r["participant_id"] for r in rows} == set(assignment), "split file has other people"
    assert_handedness_balanced(assignment, load_handedness())
    print("OK: zero participant overlap between train, val and test; both hands in val and test")
    print(summarize(assignment, rows))


if __name__ == "__main__":
    main()
