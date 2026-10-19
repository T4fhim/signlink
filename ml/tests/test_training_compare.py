"""Side-by-side table of training runs (Phase 1 step 4). Runs are ranked on val, never on test."""

import json
from pathlib import Path

import pytest
from signlnk_ml.training.compare import comparison_table, load_run


def write_run(
    folder: Path, val_top1: float, test_top1: float = 0.5, overrides: list[str] | None = None
) -> Path:
    folder.mkdir(parents=True)
    payload = {
        "run": {"n_params": 1_000_000, "epochs": 30, "overrides": overrides or []},
        "train_top1": 0.9,
        "val": {"top1": val_top1, "top5": 0.8, "min_participant_top1": 0.5},
        "test": {"top1": test_top1, "top5": 0.7, "min_participant_top1": 0.25},
    }
    (folder / "report.json").write_text(json.dumps(payload), encoding="utf-8")
    return folder


def test_load_run_reads_the_numbers_and_names_the_run_after_its_folder(tmp_path: Path) -> None:
    run = load_run(write_run(tmp_path / "velocity", 0.6, 0.55, ["input.velocity=true"]))
    assert run["name"] == "velocity" and run["test_top1"] == 0.55
    assert run["val_top1"] == 0.6 and run["train_top1"] == 0.9
    assert run["worst_test_signer"] == 0.25 and run["overrides"] == "input.velocity=true"


def test_table_is_ranked_on_val_not_test_and_marks_the_best_val_run(tmp_path: Path) -> None:
    runs = [
        load_run(write_run(tmp_path / "a", val_top1=0.50, test_top1=0.90)),
        load_run(write_run(tmp_path / "b", val_top1=0.62, test_top1=0.30)),
        load_run(write_run(tmp_path / "c", val_top1=0.55, test_top1=0.40)),
    ]
    lines = comparison_table(runs).splitlines()
    names = [line.split("|")[1].strip() for line in lines[2:]]
    assert names == ["b", "c", "a"]  # a has the best test top-1 but the worst val top-1
    assert lines[2].rstrip().endswith("best on val")
    assert "best on val" not in lines[3] and "best on val" not in lines[4]


def test_a_run_without_a_report_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "empty")
