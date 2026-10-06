"""Baseline recognizer: 1D conv stem + Transformer encoder -> 256-d embedding -> classifier.

Input is [B, T, F] from `features.make_input` (coordinates, then one 1/0 presence value per
landmark). A frame whose values are all zero has no landmark at all: it is ignored by attention and
pooling, so padding and missing frames do not change the result. The embedding is what Studio
prototypes will use later (PLAN §6.4); the classifier head covers the base vocabulary.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as torch_f
from torch import Tensor, nn

from signlnk_ml.training.config import ModelConfig

__all__ = ["BatchSource", "ModelConfig", "SignModel", "TensorSource", "fit", "fit_sources"]


class SignModel(nn.Module):
    def __init__(self, config: ModelConfig, in_dim: int, n_classes: int, max_frames: int) -> None:
        super().__init__()
        d = config.d_model
        self.conv1 = nn.Conv1d(in_dim, d, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(d, d, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(d)
        self.position = nn.Parameter(torch.randn(1, max_frames, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=4 * d,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.n_layers, enable_nested_tensor=False)
        self.embed = nn.Linear(d, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.embedding_dim, n_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (logits [B, n_classes], embedding [B, embedding_dim])."""
        frames = x.shape[1]
        empty = x.abs().sum(dim=-1) == 0  # [B, T]: no landmark detected in this frame
        keep = (~empty).to(x.dtype).unsqueeze(-1)  # [B, T, 1]
        # Empty frames are zeroed after each conv (biases would otherwise leak into neighbours).
        h = torch_f.gelu(self.conv1(x.transpose(1, 2)).transpose(1, 2)) * keep
        h = torch_f.gelu(self.conv2(h.transpose(1, 2)).transpose(1, 2)) * keep
        h = self.norm(h) + self.position[:, :frames]
        # A window with no landmarks at all attends everywhere so the outputs stay finite.
        ignore = empty & ~empty.all(dim=1, keepdim=True)
        h = self.encoder(h, src_key_padding_mask=ignore)
        pooled = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        embedding = self.embed(pooled)
        return self.head(self.dropout(embedding)), embedding


class BatchSource(Protocol):
    def __len__(self) -> int: ...

    def batch(self, indices: np.ndarray[Any, Any]) -> tuple[Tensor, Tensor]: ...


class TensorSource:
    def __init__(self, x: Tensor, y: Tensor) -> None:
        self.x, self.y = x, y

    def __len__(self) -> int:
        return len(self.y)

    def batch(self, indices: np.ndarray[Any, Any]) -> tuple[Tensor, Tensor]:
        index = torch.as_tensor(indices)
        return self.x[index], self.y[index]


def evaluate_top1(model: SignModel, source: BatchSource, batch_size: int = 512) -> float:
    model.eval()
    hits = 0
    with torch.no_grad():
        device = next(model.parameters()).device
        for start in range(0, len(source), batch_size):
            x, y = source.batch(np.arange(start, min(start + batch_size, len(source))))
            hits += int((model(x.to(device))[0].argmax(dim=1).cpu() == y).sum())
    return hits / max(len(source), 1)


def pick_device() -> torch.device:
    """The GPU when there is one (Kaggle/Colab), else the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_sources(  # noqa: PLR0913
    model: SignModel,
    train: BatchSource,
    val: BatchSource,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.01,
    label_smoothing: float = 0.0,
    seed: int = 0,
) -> list[dict[str, float]]:
    """AdamW + cosine schedule. Returns one {epoch, train_loss, val_top1} dict per epoch."""
    torch.manual_seed(seed)
    order = np.random.default_rng(seed)
    device = pick_device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * -(-len(train) // batch_size)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        permutation = order.permutation(len(train))
        total, seen = 0.0, 0
        for start in range(0, len(train), batch_size):
            x, y = (t.to(device) for t in train.batch(permutation[start : start + batch_size]))
            loss = torch_f.cross_entropy(model(x)[0], y, label_smoothing=label_smoothing)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            schedule.step()
            total, seen = total + float(loss.detach()) * len(y), seen + len(y)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / seen,
                "val_top1": evaluate_top1(model, val),
            }
        )
    return history


def fit(
    model: SignModel,
    train_x: Tensor,
    train_y: Tensor,
    val_x: Tensor,
    val_y: Tensor,
    **kwargs: Any,
) -> list[dict[str, float]]:
    """`fit_sources` for in-memory tensors."""
    return fit_sources(model, TensorSource(train_x, train_y), TensorSource(val_x, val_y), **kwargs)
