"""Regenerate the tiny ONNX fixture shared by the Python and onnxruntime-web parity tests.

    uv run python -m signlnk_ml.export.golden_fixtures

Writes `tests/fixtures/onnx/tiny.onnx` (a small seeded model, not a trained one) and
`tests/fixtures/onnx/tiny.parity.json` (inputs and PyTorch's outputs). Regenerate only for an
intended change to the exported graph, and say so: both test suites compare against these files.
"""

from __future__ import annotations

import torch

from signlnk_ml.export.onnx_export import (
    export_onnx,
    parity_report,
    synthetic_windows,
    write_parity_file,
)
from signlnk_ml.features.normalize import REPO_ROOT
from signlnk_ml.training.config import ModelConfig
from signlnk_ml.training.model import SignModel

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "onnx"
LENGTH, IN_DIM, N_CLASSES = 64, 146 * 3, 11
TINY = ModelConfig(d_model=16, n_layers=1, n_heads=2, embedding_dim=8, dropout=0.0)


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = SignModel(TINY, IN_DIM, N_CLASSES, LENGTH).eval()
    export_onnx(model, IN_DIM, LENGTH, FIXTURES / "tiny.onnx")
    windows = synthetic_windows(4, LENGTH, IN_DIM, seed=7)
    with torch.no_grad():
        logits, embedding = (t.numpy() for t in model(torch.from_numpy(windows)))
    write_parity_file(FIXTURES / "tiny.parity.json", windows, logits, embedding)
    report = parity_report(FIXTURES / "tiny.onnx", model, windows)
    print(f"tiny fixture written; parity max |logit diff| {report['max_abs_logits']:.2e}")


if __name__ == "__main__":
    main()
