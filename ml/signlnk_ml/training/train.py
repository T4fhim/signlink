"""Train the baseline recognizer and write the signer-independent report (Phase 1 step 3).

    uv run python -m signlnk_ml.training.train --config ml/configs/baseline.yaml \\
        --cache-dir <dir> --out-dir <dir> [--data-dir <kaggle-islr dir>] [--sample] [--epochs N]

If `--cache-dir` has no windows yet they are built first (see `cache`). The model is trained on the
train participants only, the final model is scored once on the val and test participants, and
`report.json` / `report.md` / `history.csv` / `model.pt` go to `--out-dir`. A `--sample` run uses
the 525-sequence local sample: it checks the pipeline and is not a result.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from signlnk_ml.data.fetch_islr import data_dir
from signlnk_ml.data.kaggle_islr import load_sign_map
from signlnk_ml.data.splits import SPLIT_NAMES, load_assignment
from signlnk_ml.features.normalize import REPO_ROOT, load_layout
from signlnk_ml.training.augment import Augmenter
from signlnk_ml.training.cache import build_split_cache, cache_mismatches, load_split, read_rows
from signlnk_ml.training.config import (
    REGISTRY_PATH,
    TrainConfig,
    check_config,
    load_config,
    load_raw,
    load_registry,
)
from signlnk_ml.training.evaluate import report
from signlnk_ml.training.features import input_dim, make_input
from signlnk_ml.training.model import SignModel, fit_sources

Windows = npt.NDArray[np.float16]


class WindowSource:
    """Cached windows as model batches; augmentation (train only) runs when a batch is fetched."""

    def __init__(
        self,
        windows: Windows,
        labels: npt.NDArray[np.int16],
        use_z: bool,
        augmenter: Augmenter | None = None,
        velocity: bool = False,
    ) -> None:
        self.windows, self.labels, self.use_z, self.augmenter = windows, labels, use_z, augmenter
        self.velocity = velocity

    def __len__(self) -> int:
        return len(self.labels)

    def batch(self, indices: npt.NDArray[Any]) -> tuple[Tensor, Tensor]:
        rows = []
        for i in indices:
            window = np.asarray(self.windows[i], dtype=np.float32)
            if self.augmenter is not None:
                window = self.augmenter(window)
            rows.append(make_input(window, self.use_z, self.velocity))
        labels = self.labels[indices].astype(np.int64)
        return torch.from_numpy(np.stack(rows)), torch.from_numpy(labels)


def predict(
    model: SignModel, source: WindowSource, batch_size: int = 512
) -> npt.NDArray[np.float32]:
    """Softmax scores [n, n_classes]."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, len(source), batch_size):
            x, _ = source.batch(np.arange(start, min(start + batch_size, len(source))))
            out.append(torch.softmax(model(x.to(device))[0], dim=1).cpu().numpy())
    return np.concatenate(out)


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def ensure_cache(cache_dir: Path, cfg: TrainConfig, root: Path, sample: bool, workers: int) -> None:
    if all((cache_dir / s / "meta.json").exists() for s in SPLIT_NAMES):
        for split in SPLIT_NAMES:
            meta = json.loads((cache_dir / split / "meta.json").read_text(encoding="utf-8"))
            if problems := cache_mismatches(meta, cfg):
                sys.exit(f"{cache_dir} was built with other window settings: {problems}")
        return
    assignment = load_assignment(REPO_ROOT / cfg.splits)
    rows = read_rows(root, sample)
    for split in SPLIT_NAMES:
        mine = [r for r in rows if assignment[r["participant_id"]] == split]
        meta = build_split_cache(root, mine, split, cache_dir / split, cfg, workers)
        print(f"cache {split}: {meta['n_signs']} sign + {meta['n_other']} other windows")


def markdown(result: dict[str, Any], run: dict[str, Any], names: dict[int, str]) -> str:
    kind = "SAMPLE: pipeline check, not a result" if run["sample"] else "full Kaggle ISLR"
    overrides = " ".join(run["overrides"]) or "none"
    lines = [
        f"# Training report ({kind})",
        "",
        f"- config: `{run['config']}` · overrides: {overrides} · commit `{run['commit']}` "
        f"· split `{run['splits']}`",
        f"- command: `{run['command']}`",
        f"- model: {run['n_params']:,} parameters, {run['epochs']} epochs, "
        f"final train loss {run['final_train_loss']:.3f}, "
        f"train top-1 {run['train_top1']:.3f} (no augmentation, {run['train_top1_n']} windows)",
        "",
    ]
    for split in ("val", "test"):
        r = result[split]
        fa = r["false_activation_rate"]
        signs_only = r["top1_signs_only"]
        lines += [
            f"## {split} signers ({len(r['per_participant_top1'])} participants, {r['n']} windows)",
            f"- top-1 {r['top1']:.3f} · top-5 {r['top5']:.3f}"
            + ("" if signs_only is None else f" · signs-only top-1 {signs_only:.3f}")
            + ("" if fa is None else f" · false activation {fa:.3f} (hands-absent windows only)"),
            f"- lowest signer top-1 {r['min_participant_top1']:.3f}; per signer: "
            + ", ".join(f"{p} {v:.3f}" for p, v in r["per_participant_top1"].items()),
            "- worst classes: "
            + ", ".join(
                f"{names[w['class']]} {w['top1']:.2f} (n={w['n']})" for w in r["worst_classes"][:5]
            ),
            "- most confused (true -> predicted): "
            + ", ".join(
                f"{names[c['true']]} -> {names[c['predicted']]} x{c['count']}"
                for c in r["top_confusions"][:5]
            ),
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "ml/configs/baseline.yaml")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set model.d_model=256 (repeatable)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    problems = check_config(load_raw(args.config, args.overrides), load_registry(REGISTRY_PATH))
    if problems:  # CLAUDE.md rule 4: a config that mixes tracks never trains
        sys.exit("config rejected:\n  " + "\n  ".join(problems))

    root = args.data_dir or data_dir()
    ensure_cache(args.cache_dir, cfg, root, args.sample, args.workers)
    splits = {s: load_split(args.cache_dir, s) for s in SPLIT_NAMES}
    n_classes, other_index = splits["train"][3]["n_classes"], splits["train"][3]["other_index"]
    layout = load_layout()
    use_z, velocity = cfg.input.use_z, cfg.input.velocity
    in_dim = input_dim(layout.n_landmarks, use_z, velocity)

    train = WindowSource(
        splits["train"][0],
        splits["train"][1],
        use_z,
        Augmenter(layout, cfg.train.seed, cfg.augment),
        velocity,
    )
    val = WindowSource(splits["val"][0], splits["val"][1], use_z, velocity=velocity)
    model = SignModel(cfg.model, in_dim, n_classes, cfg.window.length)
    epochs = args.epochs or cfg.train.epochs
    history = fit_sources(
        model,
        train,
        val,
        epochs=epochs,
        batch_size=cfg.train.batch_size,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        label_smoothing=cfg.train.label_smoothing,
        seed=cfg.train.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_top1"])
        writer.writeheader()
        writer.writerows(history)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.out_dir / "model.pt")

    result: dict[str, Any] = {}
    for split in ("val", "test"):
        windows, labels, participants, _ = splits[split]
        scores = predict(model, WindowSource(windows, labels, use_z, velocity=velocity))
        result[split] = report(scores, labels, participants, other_index)

    # Train accuracy on a fixed random subset, without augmentation: separates "does not fit"
    # (underfitting) from "fits but does not transfer to new signers" (overfitting).
    train_windows, train_labels = splits["train"][0], splits["train"][1]
    subset = np.sort(
        np.random.default_rng(0).choice(len(train_labels), min(20000, len(train_labels)), False)
    )
    train_eval = WindowSource(train_windows[subset], train_labels[subset], use_z, velocity=velocity)
    train_top1 = float((predict(model, train_eval).argmax(axis=1) == train_labels[subset]).mean())
    names = {index: sign for sign, index in load_sign_map(root).items()}
    names[other_index] = "<other>"
    config_name = (
        args.config.relative_to(REPO_ROOT) if args.config.is_relative_to(REPO_ROOT) else args.config
    )
    run = {
        "config": str(config_name),
        "commit": git_commit(),
        "splits": cfg.splits,
        "command": "python -m signlnk_ml.training.train " + " ".join(sys.argv[1:]),
        "sample": args.sample,
        "epochs": epochs,
        "n_params": sum(p.numel() for p in model.parameters()),
        "final_train_loss": history[-1]["train_loss"],
        "train_top1": train_top1,
        "train_top1_n": len(subset),
        "overrides": args.overrides,
        "track": cfg.track,
        "config_values": dataclasses.asdict(cfg),
    }
    payload = json.dumps({"run": run, "train_top1": train_top1, **result}, indent=2) + "\n"
    (args.out_dir / "report.json").write_text(payload, encoding="utf-8")
    text = markdown(result, run, names)
    (args.out_dir / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
