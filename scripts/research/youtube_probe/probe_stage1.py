"""Stage 1 of the YouTube probe (research track): decoder-style, label-free segmentation.

Port of packages/recognizer/src/decoder/endpointer.ts (DEFAULT_ENDPOINTER) finds sign segments from
hand motion alone. Each segment is resampled to 64 frames and scored once by the release ONNX model,
as the live decoder would. Captions are used only afterwards, to say which segment is meant to be
which sign (an event matches a caption anchor if it overlaps [t - 1.0 s, t + 2.5 s] the most).
Run: SIGNLNK_DATA_DIR=... uv run --no-sync python scripts/research/youtube_probe/probe_stage1.py
Gloss-to-class mappings are the assistant's reading, [VERIFY] by a fluent signer.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
from signlnk_ml.features.aspect import to_reference_aspect
from signlnk_ml.features.normalize import load_layout, normalize
from signlnk_ml.training.features import make_input, resample_nearest

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ["SIGNLNK_DATA_DIR"]) / "research-videos"
MODEL = ROOT / "apps" / "web" / "public" / "model"

# endpointer.ts DEFAULT_ENDPOINTER
START_SPEED, STOP_SPEED, START_MS, STOP_MS = 0.6, 0.3, 90, 200
PREROLL_MS, TAIL_MS, MIN_SIGN_MS, MAX_SIGN_MS, SMOOTHING = 150, 100, 250, 4000, 0.5
HAND_LANDMARKS = 42  # hand_left + hand_right come first in slk-landmarks-v1

VIDEOS = {
    "0FcwzMq4iWg": {
        "licence": "standard YouTube",
        "captions": "video.en.json3",
        "mode": "caps",
        "map": {
            "HELLO": "hello",
            "FATHER": "dad",
            "MOTHER": "mom",
            "YES": "yes",
            "NO": "no",
            "PLEASE": "please",
            "THANK YOU": "thankyou",
            "DOG": "dog",
            "CAT": "cat",
            "EAT FOOD": "food",
            "MILK": "milk",
            "GO-TO": "go",
            "BATHROOM": "potty",
            "FINE": "fine",
            "LIKE": "like",
            "FINISH": "finish",
        },
    },
    "3hyOh9UD4jw": {
        "licence": "CC BY (as labelled on YouTube)",
        "captions": "video.en-0ScBH-1_ApU.json3",
        "mode": "lines",
        "map": {
            "mother": "mom",
            "mom": "mom",
            "father": "dad",
            "dad": "dad",
            "boy": "boy",
            "girl": "girl",
            "brother": "brother",
            "grandmother": "grandma",
            "grandfather": "grandpa",
            "aunt": "aunt",
            "uncle": "uncle",
        },
    },
    "csqLvE2NeBU": {
        "licence": "CC BY (as labelled on YouTube)",
        "captions": "video.en-orig.json3",
        "mode": "words",
        "map": {
            c: c for c in ("blue", "green", "yellow", "red", "black", "brown", "white", "orange")
        },
    },
}


def hand_speed(seq: np.ndarray, fps: float) -> np.ndarray:
    """Mean x, y displacement of present hand landmarks per second (0 where none is shared)."""
    hands = seq[:, :HAND_LANDMARKS, :2]
    delta = np.hypot(*(hands[1:] - hands[:-1]).transpose(2, 0, 1))
    ok = np.isfinite(delta)
    total = np.where(ok, delta, 0.0).sum(axis=1)
    count = ok.sum(axis=1)
    speed = np.where(count > 0, total / np.maximum(count, 1) * fps, 0.0)
    return np.concatenate([[0.0], speed])


def segments(seq: np.ndarray, fps: float) -> list[tuple[int, int, int, int]]:
    """(first kept, last kept, first moving, last moving) frame indices, as Endpointer.push."""
    raw = hand_speed(seq, fps)
    ms = np.arange(len(seq)) * 1000.0 / fps
    out: list[tuple[int, int, int, int]] = []
    speed: float | None = None
    above: int | None = None
    opened: tuple[int, int] | None = None  # (first moving, last active)
    buf_start = 0
    for i, r in enumerate(raw):
        speed = r if speed is None else speed + SMOOTHING * (r - speed)
        if opened is None:
            above = (i if above is None else above) if speed >= START_SPEED else None
            if above is not None and ms[i] - ms[above] >= START_MS:
                opened = (above, i)
            continue
        first, last = opened
        if speed >= STOP_SPEED:
            last = i
            opened = (first, last)
        closing = ms[i] - ms[last] >= STOP_MS or ms[i] - ms[first] >= MAX_SIGN_MS
        if not closing:
            continue
        lo = max(buf_start, int(np.searchsorted(ms, ms[first] - PREROLL_MS)))
        hi = int(np.searchsorted(ms, ms[last] + TAIL_MS, side="right")) - 1
        if ms[last] - ms[first] >= MIN_SIGN_MS and hi >= lo:
            out.append((lo, hi, first, last))
        buf_start, above, opened = i + 1, None, None
    return out


def anchors(spec: dict, caption_file: Path) -> list[tuple[str, float]]:
    """(class, seconds) where the caption says a mapped sign; same class within 1.5 s merged."""
    events = json.loads(caption_file.read_text(encoding="utf8"))["events"]
    found: list[tuple[str, float]] = []
    for e in events:
        segs = e.get("segs", [])
        text = "".join(s.get("utf8", "") for s in segs).replace("\n", " ").strip()
        t0 = e["tStartMs"] / 1000
        if spec["mode"] == "caps":
            for gloss, cls in spec["map"].items():
                if re.search(rf"(?<![A-Z-]){re.escape(gloss)}(?![A-Z-])", text):
                    found.append((cls, t0))
        elif spec["mode"] == "lines":
            tokens = re.findall(r"[a-z]+", text.lower())
            classes = {spec["map"][w] for w in tokens if w in spec["map"]}
            if len(tokens) <= 4 and len(classes) == 1:
                found.append((classes.pop(), t0))
        else:
            for s in segs:
                word = re.sub(r"[^a-z]", "", s.get("utf8", "").lower())
                if word in spec["map"]:
                    found.append((spec["map"][word], t0 + s.get("tOffsetMs", 0) / 1000))
    found.sort(key=lambda a: a[1])
    merged: list[tuple[str, float]] = []
    for cls, t in found:
        if not any(c == cls and abs(t - u) < 1.5 for c, u in merged):
            merged.append((cls, t))
    return merged


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main() -> None:
    spec_m = json.loads((MODEL / "model.json").read_text())
    classes, length, use_z = spec_m["classes"], spec_m["input"]["length"], spec_m["input"]["use_z"]
    sess = ort.InferenceSession(str(MODEL / "model.onnx"), providers=["CPUExecutionProvider"])
    layout = load_layout()
    report: dict[str, dict] = {}
    for vid, spec in VIDEOS.items():
        run = DATA / vid / "full"
        meta = json.loads((run / "meta.json").read_text())
        fps = meta["fps"]
        raw = np.load(run / "landmarks.npy")
        seq = normalize(to_reference_aspect(raw, meta["width"], meta["height"], layout), layout)
        segs = segments(seq, fps)
        batch = np.stack(
            [make_input(resample_nearest(seq[lo : hi + 1], length), use_z) for lo, hi, _, _ in segs]
        ).astype(np.float32)
        prob = softmax(sess.run(["logits"], {"windows": batch})[0])[:, :250]
        target_ids = sorted({classes.index(c) for c in spec["map"].values()})
        rows = []
        for cls, t in anchors(spec, DATA / vid / spec["captions"]):
            best, best_ov = None, 0.0
            for k, (_, _, first, last) in enumerate(segs):
                ov = min(last / fps, t + 2.5) - max(first / fps, t - 1.0)
                if ov > best_ov:
                    best, best_ov = k, ov
            row = {"class": cls, "t": round(t, 1), "matched": best is not None}
            if best is not None:
                order = np.argsort(-prob[best])
                ci = classes.index(cls)
                row["rank_250"] = int(np.where(order == ci)[0][0]) + 1
                masked = prob[best][target_ids]
                row["rank_masked"] = int((masked > prob[best][ci]).sum()) + 1
                row["pred"] = classes[int(order[0])]
            rows.append(row)
        hit = [r for r in rows if r["matched"]]
        n = len(hit)
        report[vid] = {
            "licence": spec["licence"],
            "frames": len(seq),
            "events": len(segs),
            "anchors": len(rows),
            "matched": n,
            "target_classes": len(target_ids),
            "top1_250": sum(r["rank_250"] == 1 for r in hit),
            "top5_250": sum(r["rank_250"] <= 5 for r in hit),
            "top1_masked": sum(r["rank_masked"] == 1 for r in hit),
            "top3_masked": sum(r["rank_masked"] <= 3 for r in hit),
            "chance_masked": round(1 / len(target_ids), 3),
            "rows": rows,
        }
        print(vid, {k: v for k, v in report[vid].items() if k != "rows"})
    total = {
        k: sum(r[k] for r in report.values())
        for k in (
            "events",
            "anchors",
            "matched",
            "top1_250",
            "top5_250",
            "top1_masked",
            "top3_masked",
        )
    }
    print("TOTAL", total)
    (DATA / "probe_stage1.json").write_text(
        json.dumps({"total": total, "videos": report}, indent=1)
    )


if __name__ == "__main__":
    main()
