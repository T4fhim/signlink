"""Export a trained recognizer to a single-file ONNX model and check it against PyTorch.

    uv run python -m signlnk_ml.export.onnx_export --run-dir <training run> --out-dir <dir>
        [--sign-map <sign_to_prediction_index_map.json>]

Writes `model.onnx` (input `windows` [batch, length, dim] float32; outputs `logits` and
`embedding`), `model.json` (classes, input spec, data track, parity numbers) and `parity.json`
(inputs and PyTorch's outputs, which the onnxruntime-web test in `packages/recognizer` checks to the
same tolerance).

Only release-track runs are exported (ADR-0003, PLAN §5.1): a research-track run, or a release run
that used research data, is refused before anything is written.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import torch

from signlnk_ml.data.fetch_islr import data_dir
from signlnk_ml.features.normalize import load_layout
from signlnk_ml.training.config import ModelConfig, check_config, load_registry
from signlnk_ml.training.features import input_dim
from signlnk_ml.training.model import SignModel

PARITY_TOLERANCE = 1e-4  # PLAN §7 Phase 1 step 5: max abs diff of logits, PyTorch vs the runtime
OPSET = 18
INPUT_NAME = "windows"
OUTPUT_NAMES = ["logits", "embedding"]
Array = npt.NDArray[np.float32]


class ExportRefused(Exception):  # noqa: N818
    """The run may not be exported (data track, rule 4)."""


def export_allowed(run: Mapping[str, Any], registry: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Reasons a training run may not be exported; empty means it may."""
    values = run.get("config_values") or {}
    raw = {
        "track": run.get("track"),
        "datasets": values.get("datasets"),
        "pretrained": values.get("pretrained"),
    }
    problems = check_config(raw, registry)
    if raw["track"] != "release":
        problems.append(f"only release-track models are exported, this run is {raw['track']!r}")
    return problems


def synthetic_windows(n: int, length: int, dim: int, seed: int) -> Array:
    """Seeded model inputs [n, length, dim] with the real layout (coordinates, then a 1/0 presence
    value per landmark), including empty frames, half-empty windows and one window with no
    landmarks at all."""
    if dim % 3 != 0:
        raise ValueError("dim must be 3 * landmarks (x, y, presence)")
    rng = np.random.default_rng(seed)
    landmarks = dim // 3
    present = (rng.random((n, length, landmarks)) > 0.2).astype(np.float32)
    coords = rng.normal(0.0, 0.5, (n, length, landmarks, 2)).astype(np.float32) * present[..., None]
    windows = np.concatenate([coords.reshape(n, length, -1), present], axis=-1)
    if n >= 2:
        windows[1, length // 4 : length // 2] = 0
    if n >= 3:
        windows[2, : length // 2] = 0
    if n >= 2:
        windows[-1] = 0
    return windows.astype(np.float32)


def export_onnx(model: SignModel, in_dim: int, length: int, path: Path) -> None:
    """Writes a single-file ONNX model with a dynamic batch dimension."""
    model.eval()
    dummy = torch.from_numpy(synthetic_windows(2, length, in_dim, seed=0))
    captured = io.StringIO()  # torch prints progress (with emoji) that cp1252 consoles cannot show
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            torch.onnx.export(
                model,
                (dummy,),
                str(path),
                input_names=[INPUT_NAME],
                output_names=OUTPUT_NAMES,
                dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
                opset_version=OPSET,
                dynamo=True,
                external_data=False,
            )
    except Exception as error:
        raise RuntimeError(f"ONNX export failed: {error}\n{captured.getvalue()[-2000:]}") from error


def onnx_outputs(path: Path, windows: Array) -> tuple[Array, Array]:
    """(logits, embedding) from onnxruntime on the CPU."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    logits, embedding = session.run(None, {INPUT_NAME: np.ascontiguousarray(windows)})
    return np.asarray(logits, dtype=np.float32), np.asarray(embedding, dtype=np.float32)


def parity_report(path: Path, model: SignModel, windows: Array) -> dict[str, Any]:
    """Max abs difference between PyTorch and onnxruntime on the same windows."""
    model.eval()
    with torch.no_grad():
        ref_logits, ref_embedding = (t.numpy() for t in model(torch.from_numpy(windows)))
    logits, embedding = onnx_outputs(path, windows)
    return {
        "n": len(windows),
        "max_abs_logits": float(np.abs(logits - ref_logits).max()),
        "max_abs_embedding": float(np.abs(embedding - ref_embedding).max()),
        "finite": bool(np.isfinite(logits).all() and np.isfinite(embedding).all()),
        "tolerance": PARITY_TOLERANCE,
    }


def _encode(array: Array) -> dict[str, Any]:
    data = np.ascontiguousarray(array, dtype="<f4")
    return {"shape": list(data.shape), "b64": base64.b64encode(data.tobytes()).decode("ascii")}


def _decode(entry: Mapping[str, Any]) -> Array:
    raw = np.frombuffer(base64.b64decode(entry["b64"]), dtype="<f4")
    return raw.reshape(entry["shape"]).astype(np.float32)


def write_parity_file(path: Path, windows: Array, logits: Array, embedding: Array) -> None:
    """Inputs plus PyTorch's outputs (little-endian float32, base64) for other runtimes to check."""
    payload = {
        "format": "slk-parity-v1",
        "tolerance": PARITY_TOLERANCE,
        "windows": _encode(windows),
        "logits": _encode(logits),
        "embedding": _encode(embedding),
    }
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload) + "\n")


def read_parity_file(path: Path) -> tuple[Array, Array, Array]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return _decode(data["windows"]), _decode(data["logits"]), _decode(data["embedding"])


def export_run(
    run_dir: Path,
    out_dir: Path,
    classes: Sequence[str],
    registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Exports `run_dir/model.pt` to `out_dir`; raises ExportRefused before writing anything."""
    run: dict[str, Any] = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))["run"]
    problems = export_allowed(run, registry if registry is not None else load_registry())
    if problems:
        raise ExportRefused("; ".join(problems))
    values = run["config_values"]
    if values.get("vocabulary", {}).get("file"):
        raise ValueError("export of a list-only model is not supported: export the full model")

    layout = load_layout()
    use_z, velocity = values["input"]["use_z"], values["input"]["velocity"]
    length = values["window"]["length"]
    in_dim = input_dim(layout.n_landmarks, use_z, velocity)
    state = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    outputs = state["head.weight"].shape[0]
    if outputs != len(classes):
        raise ValueError(f"classes has {len(classes)} entries but the model has {outputs} outputs")
    model = SignModel(ModelConfig(**values["model"]), in_dim, len(classes), length)
    model.load_state_dict(state)
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    export_onnx(model, in_dim, length, onnx_path)
    windows = synthetic_windows(8, length, in_dim, seed=1)
    report = parity_report(onnx_path, model, windows)
    with torch.no_grad():
        ref_logits, ref_embedding = (t.numpy() for t in model(torch.from_numpy(windows)))
    write_parity_file(out_dir / "parity.json", windows, ref_logits, ref_embedding)
    meta = {
        "format": "slk-recognizer-v1",
        "classes": list(classes),
        "other_index": len(classes) - 1,
        "input": {
            "name": INPUT_NAME,
            "length": length,
            "dim": in_dim,
            "use_z": use_z,
            "velocity": velocity,
            "n_landmarks": layout.n_landmarks,
            "layout": layout.layout_id,
        },
        "outputs": OUTPUT_NAMES,
        "opset": OPSET,
        "track": run["track"],
        "datasets": values["datasets"],
        "pretrained": values.get("pretrained", []),
        "source": {"commit": run.get("commit"), "command": run.get("command")},
        "parity": report,
    }
    with (out_dir / "model.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(meta, indent=2) + "\n")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--sign-map",
        type=Path,
        default=data_dir() / "sign_to_prediction_index_map.json",
        help="sign name -> class index (Kaggle ISLR file)",
    )
    args = parser.parse_args()

    sign_map: dict[str, int] = json.loads(args.sign_map.read_text(encoding="utf-8"))
    classes = [*sorted(sign_map, key=lambda sign: sign_map[sign]), "<other>"]
    try:
        out = export_run(args.run_dir, args.out_dir, classes)
    except ExportRefused as refusal:
        sys.exit(f"export refused: {refusal}")
    meta = json.loads((out / "model.json").read_text(encoding="utf-8"))
    parity = meta["parity"]
    size = (out / "model.onnx").stat().st_size / 1e6
    print(f"wrote {out} ({size:.2f} MB model, {len(classes)} classes)")
    print(
        f"parity on {parity['n']} windows: max |logit diff| {parity['max_abs_logits']:.2e}, "
        f"max |embedding diff| {parity['max_abs_embedding']:.2e} (tolerance {PARITY_TOLERANCE:g})"
    )
    if (
        parity["max_abs_logits"] > PARITY_TOLERANCE
        or parity["max_abs_embedding"] > PARITY_TOLERANCE
    ):
        sys.exit("parity check FAILED")


if __name__ == "__main__":
    main()
