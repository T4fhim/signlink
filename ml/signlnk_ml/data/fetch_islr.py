"""Dev-time download of a participant-balanced sample of Kaggle ISLR (Google asl-signs).

Run: uv run python -m signlnk_ml.data.fetch_islr --per-participant 25

Needs the `kaggle` CLI on PATH (`uv tool install kaggle`), a token in ~/.kaggle/access_token, and
the competition rules accepted on kaggle.com. The CLI is a dev tool, not a project dependency. The
full dataset is ~40 GB; only train.csv, the sign index map and the sampled sequences are fetched.

Data goes to $SIGNLNK_DATA_DIR/kaggle-islr (default: <repo>/data, which is gitignored).
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from signlnk_ml.features.normalize import REPO_ROOT

COMPETITION = "asl-signs"


@dataclass(frozen=True)
class SequenceRow:
    """One line of train.csv."""

    path: str
    participant_id: str
    sequence_id: str
    sign: str


def data_root() -> Path:
    """Machine-local data directory: $SIGNLNK_DATA_DIR, else <repo>/data (gitignored)."""
    return Path(os.environ.get("SIGNLNK_DATA_DIR", REPO_ROOT / "data"))


def data_dir() -> Path:
    """Root of the Kaggle ISLR copy on this machine."""
    return data_root() / "kaggle-islr"


def read_train_table(root: Path) -> list[SequenceRow]:
    with (root / "train.csv").open(newline="", encoding="utf-8") as handle:
        return [
            SequenceRow(r["path"], r["participant_id"], r["sequence_id"], r["sign"])
            for r in csv.DictReader(handle)
        ]


def select_sample(rows: list[SequenceRow], per_participant: int) -> list[SequenceRow]:
    """Deterministic, evenly spread over each participant's signs (no RNG)."""
    by_participant: dict[str, list[SequenceRow]] = {}
    for row in rows:
        by_participant.setdefault(row.participant_id, []).append(row)
    chosen: list[SequenceRow] = []
    for participant in sorted(by_participant):
        ordered = sorted(by_participant[participant], key=lambda r: (r.sign, r.sequence_id))
        count = min(per_participant, len(ordered))
        chosen.extend(ordered[(i * len(ordered)) // count] for i in range(count))
    return chosen


def _present(path: Path) -> bool:
    """A file counts as downloaded only if non-empty (an interrupted download leaves 0 bytes)."""
    return path.exists() and path.stat().st_size > 0


def _download(root: Path, relative: str) -> str | None:
    """Fetches one file with the kaggle CLI. Returns an error message, or None on success."""
    target_dir = root / Path(relative).parent
    target_dir.mkdir(parents=True, exist_ok=True)
    if _present(root / relative):
        return None
    result = subprocess.run(
        ["kaggle", "competitions", "download", COMPETITION, "-f", relative, "-p", str(target_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    archive = target_dir / f"{Path(relative).name}.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(target_dir)
        archive.unlink()
    if not _present(root / relative):
        return f"{relative}: {result.stderr.strip() or result.stdout.strip()}"
    return None


def fetch(root: Path, per_participant: int, workers: int) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    errors = [
        e
        for name in ("train.csv", "sign_to_prediction_index_map.json")
        if (e := _download(root, name))
    ]
    if errors:
        return errors
    sample = select_sample(read_train_table(root), per_participant)
    with (root / "sample_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "participant_id", "sequence_id", "sign"])
        writer.writerows((r.path, r.participant_id, r.sequence_id, r.sign) for r in sample)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda row: _download(root, row.path), sample))
    return [e for e in results if e]


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--per-participant", type=int, default=25)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = data_dir()
    print(f"fetching into {root}")
    errors = fetch(root, args.per_participant, args.workers)
    for error in errors[:10]:
        print("FAILED", error)
    print(f"done, {len(errors)} failures")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
