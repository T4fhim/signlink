"""Leave-one-video-out fine-tune experiment (research track, CPU, minutes).

For each held-out video: fine-tune the Kaggle-only baseline on clips cut from the other videos
(plus Kaggle sample windows as replay) and score it on the held-out video's caption-anchored
segments, the same ones `probe_stage1.py` scores. A control fine-tunes on the replay alone with the
same settings, so the effect of the video clips is separated from "a few more epochs". Clip labels
are the captions' (unverified); results are per caption anchor, small and noisy: a feasibility
signal, not accuracy.

Run: SIGNLNK_DATA_DIR=... uv run --no-sync python scripts/research/youtube_probe/finetune_loo.py
Config: ml/configs/research-youtube-finetune.yaml (track: research). Models are not saved.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from probe_stage1 import DATA, MODEL, ROOT, VIDEOS, anchors, segments
from signlnk_ml.features.aspect import to_reference_aspect
from signlnk_ml.features.normalize import load_layout, normalize
from signlnk_ml.training.augment import Augmenter
from signlnk_ml.training.cache import load_split
from signlnk_ml.training.config import (
    REGISTRY_PATH,
    check_config,
    load_config,
    load_raw,
    load_registry,
)
from signlnk_ml.training.features import input_dim, resample_nearest
from signlnk_ml.training.model import SignModel, fit_sources
from signlnk_ml.training.train import WindowSource, predict

CONFIG = ROOT / "ml" / "configs" / "research-youtube-finetune.yaml"
KAGGLE = Path(os.environ["SIGNLNK_DATA_DIR"]) / "kaggle-islr"


def video_clips(vid: str, classes: list[str], length: int, layout) -> list[dict]:
    """Caption-anchored segments of one video as windows [length, N, 3] with class indices."""
    spec = VIDEOS[vid]
    run = DATA / vid / "full"
    meta = json.loads((run / "meta.json").read_text())
    fps = meta["fps"]
    raw = np.load(run / "landmarks.npy")
    seq = normalize(to_reference_aspect(raw, meta["width"], meta["height"], layout), layout)
    segs = segments(seq, fps)
    clips = []
    for cls, t in anchors(spec, DATA / vid / spec["captions"]):
        best, best_ov = None, 0.0
        for k, (_, _, first, last) in enumerate(segs):
            ov = min(last / fps, t + 2.5) - max(first / fps, t - 1.0)
            if ov > best_ov:
                best, best_ov = k, ov
        if best is not None:
            lo, hi = segs[best][:2]
            window = resample_nearest(seq[lo : hi + 1], length).astype(np.float32)
            clips.append({"window": window, "label": classes.index(cls), "class": cls})
    return clips


def score(model: SignModel, clips: list[dict], target_ids: list[int]) -> dict:
    windows = np.stack([c["window"] for c in clips])
    labels = np.array([c["label"] for c in clips], dtype=np.int16)
    prob = predict(model, WindowSource(windows, labels, False))[:, :250]
    top1 = top5 = m1 = 0
    for p, y in zip(prob, labels, strict=True):
        rank = int((p > p[y]).sum()) + 1
        top1 += rank == 1
        top5 += rank <= 5
        m1 += int((p[target_ids] > p[y]).sum()) == 0
    return {"n": len(clips), "top1_250": int(top1), "top5_250": int(top5), "top1_masked": int(m1)}


def finetune(base, cfg, windows, labels, val, seed, ft):  # noqa: PLR0913, PLR0917
    model = copy.deepcopy(base)
    source = WindowSource(windows, labels, False, Augmenter(load_layout(), seed, cfg.augment))
    fit_sources(
        model,
        source,
        val,
        epochs=ft["epochs"],
        batch_size=ft["batch_size"],
        lr=ft["lr"],
        weight_decay=ft["weight_decay"],
        label_smoothing=ft["label_smoothing"],
        seed=seed,
    )
    return model


def mean(rows: list[dict], key: str) -> float:
    return round(float(np.mean([r[key] for r in rows])), 1)


def main() -> None:
    raw_cfg = load_raw(CONFIG)
    problems = check_config(raw_cfg, load_registry(REGISTRY_PATH))
    if problems:
        raise SystemExit("config rejected:\n  " + "\n  ".join(problems))
    ft = raw_cfg["finetune"]
    cfg = load_config(ROOT / "ml" / "configs" / "baseline.yaml")
    layout = load_layout()
    spec_m = json.loads((MODEL / "model.json").read_text())
    classes, length = spec_m["classes"], spec_m["input"]["length"]

    cache = KAGGLE / raw_cfg["cache"]
    tr_w, tr_y, _, tr_meta = load_split(cache, "train")
    va_w, va_y, _, _ = load_split(cache, "val")
    assert tr_meta["signs"] == classes[:250], "cache class order differs from the model's"
    replay_w, replay_y = np.asarray(tr_w, dtype=np.float32), np.asarray(tr_y)
    val = WindowSource(np.asarray(va_w, dtype=np.float32), va_y, False)

    base = SignModel(cfg.model, input_dim(layout.n_landmarks, False), 251, length)
    state = torch.load(
        KAGGLE / raw_cfg["base_run"] / "model.pt", map_location="cpu", weights_only=True
    )
    base.load_state_dict(state)

    clips = {vid: video_clips(vid, classes, length, layout) for vid in VIDEOS}
    out: dict[str, dict] = {}
    for held in VIDEOS:
        target_ids = sorted({classes.index(c) for c in VIDEOS[held]["map"].values()})
        train_clips = [c for vid, cs in clips.items() if vid != held for c in cs]
        cw = np.stack([c["window"] for c in train_clips] * ft["clip_repeat"])
        cl = np.array([c["label"] for c in train_clips] * ft["clip_repeat"], dtype=np.int16)
        mixed_w, mixed_y = np.concatenate([replay_w, cw]), np.concatenate([replay_y, cl])
        fold = {
            "held_out_anchors": len(clips[held]),
            "train_clips": len(train_clips),
            "base": score(base, clips[held], target_ids),
            "base_kaggle_val_top1": float((predict(base, val).argmax(1) == va_y).mean()),
            "control": [],
            "finetuned": [],
            "finetuned_kaggle_val_top1": [],
        }
        for seed in ft["seeds"]:
            control = finetune(base, cfg, replay_w, replay_y, val, seed, ft)
            tuned = finetune(base, cfg, mixed_w, mixed_y, val, seed, ft)
            fold["control"].append(score(control, clips[held], target_ids))
            fold["finetuned"].append(score(tuned, clips[held], target_ids))
            fold["finetuned_kaggle_val_top1"].append(
                float((predict(tuned, val).argmax(1) == va_y).mean())
            )
        out[held] = fold
        keys = ("top1_250", "top5_250", "top1_masked")
        print(
            held,
            "n =",
            fold["held_out_anchors"],
            "| base",
            {k: fold["base"][k] for k in keys},
            "| control",
            {k: mean(fold["control"], k) for k in keys},
            "| finetuned",
            {k: mean(fold["finetuned"], k) for k in keys},
            "| kaggle val top1",
            round(fold["base_kaggle_val_top1"], 3),
            "->",
            round(float(np.mean(fold["finetuned_kaggle_val_top1"])), 3),
            flush=True,
        )
    (DATA / "finetune_loo.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
