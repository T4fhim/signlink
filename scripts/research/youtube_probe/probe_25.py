"""Zero-shot domain-gap probe (research track): current release ONNX model on landmarks extracted
from one tutorial video. Run with: uv run --no-sync python probe_25.py (from C:\\dev\\signlnk).

Segments come from the video's own caption timestamps (gloss shown in capitals). Class mapping is
the assistant's reading of the gloss against the model's class list: [VERIFY] by a fluent signer.
Prediction per segment uses no label: peak softmax per class across all sliding windows with hands.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from signlnk_ml.features.aspect import to_reference_aspect
from signlnk_ml.features.normalize import load_layout, normalize
from signlnk_ml.training.features import make_input, resample_nearest

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(os.environ["SIGNLNK_DATA_DIR"]) / "research-videos" / "0FcwzMq4iWg" / "full"
MODEL = ROOT / "apps" / "web" / "public" / "model"

meta = json.loads((RUN / "meta.json").read_text())
fps, W, H = meta["fps"], meta["width"], meta["height"]
layout = load_layout()
raw = np.load(RUN / "landmarks.npy")
seq = normalize(to_reference_aspect(raw, W, H, layout), layout)

spec = json.loads((MODEL / "model.json").read_text())
classes = spec["classes"]
length, use_z = spec["input"]["length"], spec["input"]["use_z"]
sess = ort.InferenceSession(str(MODEL / "model.onnx"), providers=["CPUExecutionProvider"])

# (gloss in video, first caption time in s, model class or None)
SIGNS = [
    ("HELLO", 78.2, "hello"),
    ("SEE YOU LATER", 85.6, None),
    ("ME", 100.0, None),
    ("FATHER", 104.1, "dad"),
    ("MOTHER", 114.7, "mom"),
    ("YES", 121.8, "yes"),
    ("NO", 132.2, "no"),
    ("HELP", 139.4, None),
    ("PLEASE", 154.2, "please"),
    ("THANK YOU", 160.2, "thankyou"),
    ("WANT", 167.7, None),
    ("WHAT", 177.1, None),
    ("DOG", 186.2, "dog"),
    ("CAT", 195.2, "cat"),
    ("AGAIN", 200.3, None),
    ("EAT FOOD", 208.2, "food"),
    ("MILK", 224.7, "milk"),
    ("MORE", 229.8, None),
    ("GO-TO", 239.4, "go"),
    ("BATHROOM", 247.5, "potty"),
    ("FINE", 257.8, "fine"),
    ("LIKE", 265.5, "like"),
    ("LEARN", 275.7, None),
    ("SIGN", 283.9, None),
    ("FINISH", 292.6, "finish"),
]
END = 301.8
hands = np.isfinite(seq[:, 0:42, 0]).any(axis=1)  # hand_left + hand_right occupy indices 0..41


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


rows = []
for k, (gloss, t0, cls) in enumerate(SIGNS):
    t1 = SIGNS[k + 1][1] if k + 1 < len(SIGNS) else END
    a, b = int(t0 * fps), int(t1 * fps)
    wins = []
    for n in (24, 36, 48, 72):
        for s in range(a, b - n, 6):
            if hands[s : s + n].mean() >= 0.5:
                wins.append(make_input(resample_nearest(seq[s : s + n], length), use_z))
    if not wins:
        rows.append({"sign": gloss, "class": cls, "windows": 0})
        continue
    p = softmax(sess.run(["logits"], {"windows": np.stack(wins).astype(np.float32)})[0])[:, :250]
    peak = p.max(axis=0)
    order = np.argsort(-peak)
    row = {
        "sign": gloss,
        "class": cls,
        "t0": t0,
        "t1": t1,
        "windows": len(wins),
        "top3_predicted": [(classes[i], round(float(peak[i]), 3)) for i in order[:3]],
    }
    if cls:
        ci = classes.index(cls)
        row["rank_250"] = int(np.where(order == ci)[0][0]) + 1
        row["peak_target"] = round(float(peak[ci]), 3)
        row["_peak"] = peak
    rows.append(row)

mapped = [r for r in rows if r.get("class") and "rank_250" in r]
mask = np.array([classes.index(r["class"]) for r in mapped])
for i, r in enumerate(mapped):
    sub = r.pop("_peak")[mask]
    r["rank_masked"] = int((sub > sub[i]).sum()) + 1
for r in rows:
    r.pop("_peak", None)

n = len(mapped)
summary = {
    "signs_in_video": len(SIGNS),
    "mapped_to_class": n,
    "unmapped": [r["sign"] for r in rows if not r.get("class")],
    "top1_250": sum(r["rank_250"] == 1 for r in mapped),
    "top5_250": sum(r["rank_250"] <= 5 for r in mapped),
    "top1_masked": sum(r["rank_masked"] == 1 for r in mapped),
    "top3_masked": sum(r["rank_masked"] <= 3 for r in mapped),
    "chance_top1_250": round(1 / 250, 4),
    "chance_top1_masked": round(1 / n, 3),
    "hands_in_view_frac": round(float(hands.mean()), 3),
}
out = {
    "summary": summary,
    "rows": rows,
    "model": spec["source"]["commit"],
    "mediapipe": meta["mediapipe"],
}
(RUN.parent / "probe.json").write_text(json.dumps(out, indent=1))
print(json.dumps(summary, indent=1))
for r in rows:
    print({k: v for k, v in r.items() if k not in ("t0", "t1")})
