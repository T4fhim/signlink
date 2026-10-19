"""ONNX export of the recognizer and its parity check (Phase 1 step 5).

Needs torch and the export extra (`uv sync --all-packages --extra train --extra export`).
"""

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from signlnk_ml.export.onnx_export import (  # noqa: E402
    PARITY_TOLERANCE,
    ExportRefused,
    export_allowed,
    export_onnx,
    export_run,
    onnx_outputs,
    parity_report,
    read_parity_file,
    synthetic_windows,
    write_parity_file,
)
from signlnk_ml.features.normalize import REPO_ROOT  # noqa: E402
from signlnk_ml.training.config import CONFIGS_DIR, ModelConfig, load_config  # noqa: E402
from signlnk_ml.training.model import SignModel  # noqa: E402

LENGTH, IN_DIM, N_CLASSES = 64, 146 * 3, 11
TINY = ModelConfig(d_model=16, n_layers=1, n_heads=2, embedding_dim=8, dropout=0.0)
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "onnx"
REGISTRY = {"kaggle-islr": {"track": "release"}, "asl-citizen": {"track": "research"}}


def tiny_model() -> "SignModel":
    torch.manual_seed(0)
    return SignModel(TINY, in_dim=IN_DIM, n_classes=N_CLASSES, max_frames=LENGTH).eval()


def test_onnxruntime_telemetry_is_switched_off_on_import() -> None:
    """Python onnxruntime sends Microsoft telemetry on Windows unless this is set (Privacy.md)."""
    assert os.environ.get("ORT_DISABLE_TELEMETRY") == "1"


def test_the_parity_tolerance_is_the_plan_value() -> None:
    assert PARITY_TOLERANCE == 1e-4  # PLAN §7 Phase 1 step 5


def test_export_writes_one_valid_file_with_named_inputs_and_outputs(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    export_onnx(tiny_model(), IN_DIM, LENGTH, path)
    assert path.exists() and [p.name for p in tmp_path.iterdir()] == ["model.onnx"]  # no .data file
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    assert [i.name for i in graph.graph.input] == ["windows"]
    assert [o.name for o in graph.graph.output] == ["logits", "embedding"]


def test_onnxruntime_matches_pytorch_within_1e_4_on_edge_windows(tmp_path: Path) -> None:
    model, path = tiny_model(), tmp_path / "model.onnx"
    export_onnx(model, IN_DIM, LENGTH, path)
    windows = synthetic_windows(6, LENGTH, IN_DIM, seed=3)  # empty frames and an empty window
    report = parity_report(path, model, windows)
    assert report["max_abs_logits"] <= PARITY_TOLERANCE
    assert report["max_abs_embedding"] <= PARITY_TOLERANCE
    assert report["finite"] is True and report["n"] == 6


def test_the_exported_model_accepts_any_batch_size(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    export_onnx(tiny_model(), IN_DIM, LENGTH, path)
    for batch in (1, 3, 9):
        logits, embedding = onnx_outputs(path, synthetic_windows(batch, LENGTH, IN_DIM, seed=batch))
        assert logits.shape == (batch, N_CLASSES) and embedding.shape == (batch, TINY.embedding_dim)


def test_synthetic_windows_are_seeded_and_cover_the_edge_cases() -> None:
    a = synthetic_windows(6, LENGTH, IN_DIM, seed=1)
    b = synthetic_windows(6, LENGTH, IN_DIM, seed=1)
    assert np.array_equal(a, b) and a.dtype == np.float32 and a.shape == (6, LENGTH, IN_DIM)
    assert not a[-1].any()  # the last window has no landmark at all
    assert (np.abs(a).sum(axis=-1) == 0).any(axis=1).sum() >= 2  # windows with some empty frames


def test_parity_file_round_trips(tmp_path: Path) -> None:
    windows = synthetic_windows(3, 8, 12, seed=0)
    logits, embedding = np.ones((3, 4), np.float32), np.zeros((3, 2), np.float32)
    path = tmp_path / "parity.json"
    write_parity_file(path, windows, logits, embedding)
    w2, l2, e2 = read_parity_file(path)
    assert np.array_equal(w2, windows) and np.array_equal(l2, logits)
    assert np.array_equal(e2, embedding)


def test_export_allowed_accepts_only_clean_release_runs() -> None:
    clean = {"track": "release", "config_values": {"datasets": ["kaggle-islr"], "pretrained": []}}
    assert export_allowed(clean, REGISTRY) == []
    research = {**clean, "track": "research"}
    assert any("research" in p for p in export_allowed(research, REGISTRY))
    mixed = {"track": "release", "config_values": {"datasets": ["kaggle-islr", "asl-citizen"]}}
    assert any("asl-citizen" in p for p in export_allowed(mixed, REGISTRY))
    unknown = {"track": "release", "config_values": {"datasets": ["mystery"]}}
    assert any("mystery" in p for p in export_allowed(unknown, REGISTRY))
    assert export_allowed({"config_values": {}}, REGISTRY)  # no track at all


def make_run(folder: Path, track: str = "release", datasets: list[str] | None = None) -> Path:
    """A fake training-run folder holding the tiny model and a matching report.json."""
    folder.mkdir()
    cfg = load_config(CONFIGS_DIR / "baseline.yaml")
    values = dataclasses.asdict(cfg)
    values["model"] = dataclasses.asdict(TINY)
    values["datasets"] = datasets or ["kaggle-islr"]
    run = {"track": track, "config_values": values, "commit": "abc123", "n_params": 1}
    (folder / "report.json").write_text(json.dumps({"run": run}), encoding="utf-8")
    torch.save(tiny_model().state_dict(), folder / "model.pt")
    return folder


CLASSES = [f"sign{i}" for i in range(N_CLASSES - 1)] + ["<other>"]


def test_export_run_writes_model_metadata_and_parity_files(tmp_path: Path) -> None:
    out = export_run(make_run(tmp_path / "run"), tmp_path / "out", CLASSES, REGISTRY)
    assert sorted(p.name for p in out.iterdir()) == ["model.json", "model.onnx", "parity.json"]
    meta = json.loads((out / "model.json").read_text(encoding="utf-8"))
    assert meta["classes"] == CLASSES and meta["other_index"] == N_CLASSES - 1
    assert meta["input"] == {
        "name": "windows",
        "length": LENGTH,
        "dim": IN_DIM,
        "use_z": False,
        "velocity": False,
        "n_landmarks": 146,
        "layout": "slk-landmarks-v1",
    }
    assert meta["outputs"] == ["logits", "embedding"] and meta["track"] == "release"
    assert meta["datasets"] == ["kaggle-islr"] and meta["source"]["commit"] == "abc123"
    assert meta["parity"]["max_abs_logits"] <= PARITY_TOLERANCE
    w, logits, _ = read_parity_file(out / "parity.json")
    assert np.abs(onnx_outputs(out / "model.onnx", w)[0] - logits).max() <= PARITY_TOLERANCE


def test_export_run_refuses_a_research_run_and_writes_nothing(tmp_path: Path) -> None:
    run = make_run(tmp_path / "run", track="research")
    with pytest.raises(ExportRefused, match="research"):
        export_run(run, tmp_path / "out", CLASSES, REGISTRY)
    assert not (tmp_path / "out").exists()


def test_export_run_refuses_research_data_in_a_release_run(tmp_path: Path) -> None:
    run = make_run(tmp_path / "run", datasets=["kaggle-islr", "asl-citizen"])
    with pytest.raises(ExportRefused, match="asl-citizen"):
        export_run(run, tmp_path / "out", CLASSES, REGISTRY)


def test_export_run_checks_the_class_list_matches_the_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="classes"):
        export_run(make_run(tmp_path / "run"), tmp_path / "out", CLASSES[:-1], REGISTRY)


def test_committed_tiny_fixture_still_matches_onnxruntime() -> None:
    """The contract the TS (onnxruntime-web) parity test reads too: tests/fixtures/onnx/."""
    windows, logits, embedding = read_parity_file(FIXTURES / "tiny.parity.json")
    out_logits, out_embedding = onnx_outputs(FIXTURES / "tiny.onnx", windows)
    assert np.abs(out_logits - logits).max() <= PARITY_TOLERANCE
    assert np.abs(out_embedding - embedding).max() <= PARITY_TOLERANCE
    assert windows.shape[1:] == (LENGTH, IN_DIM) and logits.shape[1] == N_CLASSES
