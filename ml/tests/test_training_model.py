"""The baseline encoder + classifier (Phase 1 step 3). Needs torch (`uv sync --extra train`)."""

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from signlnk_ml.training.model import ModelConfig, SignModel, fit  # noqa: E402

N_LANDMARKS = 146
COORDS = N_LANDMARKS * 2  # use_z=False: x, y per landmark, then one mask value per landmark
IN_DIM = COORDS + N_LANDMARKS


def inputs(batch: int, frames: int, seed: int = 0) -> Any:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, frames, IN_DIM, generator=generator)
    x[..., COORDS:] = 1.0  # all landmarks present
    return x


def small(**overrides: int) -> ModelConfig:
    base = {"d_model": 32, "n_layers": 1, "n_heads": 2, "embedding_dim": 16, "dropout": 0.0}
    return ModelConfig(**(base | overrides))  # type: ignore[arg-type]


def test_outputs_have_the_expected_shapes() -> None:
    model = SignModel(small(), in_dim=IN_DIM, n_classes=11, max_frames=16)
    logits, embedding = model(inputs(4, 16))
    assert logits.shape == (4, 11) and embedding.shape == (4, 16)


def test_default_embedding_is_256_dimensional_and_the_model_is_small() -> None:
    model = SignModel(ModelConfig(), in_dim=IN_DIM, n_classes=251, max_frames=64)
    assert ModelConfig().embedding_dim == 256
    assert sum(p.numel() for p in model.parameters()) < 5_000_000


def test_a_window_with_no_landmarks_gives_finite_outputs() -> None:
    model = SignModel(small(), in_dim=IN_DIM, n_classes=5, max_frames=8).eval()
    logits, embedding = model(torch.zeros(2, 8, IN_DIM))  # mask all zero: nothing detected
    assert torch.isfinite(logits).all() and torch.isfinite(embedding).all()


def test_padding_frames_do_not_change_the_result() -> None:
    """Frames with no landmark present are ignored, so appended empty frames change nothing."""
    model = SignModel(small(), in_dim=IN_DIM, n_classes=5, max_frames=16).eval()
    x = inputs(2, 8)
    padded = torch.cat([x, torch.zeros(2, 8, IN_DIM)], dim=1)
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(padded)
    assert torch.allclose(a, b, atol=1e-4)


def test_fit_memorises_a_tiny_dataset() -> None:
    torch.manual_seed(0)
    x = inputs(12, 8, seed=3)
    y = torch.arange(12) % 3
    model = SignModel(small(), in_dim=IN_DIM, n_classes=3, max_frames=8)
    history = fit(model, x, y, x, y, epochs=60, batch_size=12, lr=3e-3, seed=0)
    assert history[-1]["train_loss"] < 0.3 * history[0]["train_loss"]
    assert history[-1]["val_top1"] == pytest.approx(1.0)


def test_fit_is_reproducible_for_a_seed() -> None:
    x, y = inputs(8, 8, seed=2), torch.arange(8) % 2

    def run() -> float:
        torch.manual_seed(0)
        model = SignModel(small(), in_dim=IN_DIM, n_classes=2, max_frames=8)
        history = fit(model, x, y, x, y, epochs=3, batch_size=4, lr=1e-3, seed=7)
        return float(history[-1]["train_loss"])

    assert run() == pytest.approx(run())
