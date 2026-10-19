"""Side-by-side table of training runs. Runs are ranked on val top-1, never on test.

    uv run python -m signlnk_ml.training.compare <run dir> [<run dir> ...]

Each run dir holds the `report.json` written by `train`. Test numbers are shown for reference only:
choosing a configuration by its test score would turn the test signers into a second validation set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

COLUMNS = (
    "run",
    "overrides",
    "params",
    "epochs",
    "train top-1",
    "val top-1",
    "val top-5",
    "test top-1",
    "test top-5",
    "worst test signer",
)


def load_run(folder: Path) -> dict[str, Any]:
    data = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    run = data["run"]
    return {
        "name": folder.name,
        "overrides": " ".join(run.get("overrides") or []),
        "n_params": run["n_params"],
        "epochs": run["epochs"],
        "train_top1": data.get("train_top1"),
        "val_top1": data["val"]["top1"],
        "val_top5": data["val"]["top5"],
        "test_top1": data["test"]["top1"],
        "test_top5": data["test"]["top5"],
        "worst_test_signer": data["test"]["min_participant_top1"],
    }


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def comparison_table(runs: list[dict[str, Any]]) -> str:
    ranked = sorted(runs, key=lambda r: -r["val_top1"])
    lines = ["| " + " | ".join((*COLUMNS, "note")), "|" + "---|" * (len(COLUMNS) + 1)]
    for rank, r in enumerate(ranked):
        cells = [
            r["name"],
            r["overrides"] or "(baseline)",
            f"{r['n_params']:,}",
            str(r["epochs"]),
            _num(r["train_top1"]),
            _num(r["val_top1"]),
            _num(r["val_top5"]),
            _num(r["test_top1"]),
            _num(r["test_top5"]),
            _num(r["worst_test_signer"]),
            "best on val" if rank == 0 else "",
        ]
        lines.append("| " + " | ".join(cells))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    print(comparison_table([load_run(folder) for folder in args.runs]))
    print("\nRanked on val; test is for reference only, never for choosing a configuration.")


if __name__ == "__main__":
    main()
