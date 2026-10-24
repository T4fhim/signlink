"""Research-track probe: video -> slk-landmarks-v1 raw frames [T, 146, 3] (NaN = missing).

Mirrors packages/landmarks/src/frame.ts (assembleFrame, hand assignment by nearer pose wrist,
ADR-0008) using the Python MediaPipe Tasks API and the repo's .task models. Raw image-normalized
coordinates, like the browser recorder; aspect and normalization are applied downstream.
Landmarks stay local under $SIGNLNK_DATA_DIR.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

REPO = Path(__file__).resolve().parents[3]
MODELS = REPO / "apps" / "web" / "public" / "mediapipe" / "models"
LAYOUT = json.loads(
    (REPO / "packages" / "schemas" / "layouts" / "slk-landmarks-v1.json").read_text()
)

video = Path(sys.argv[1])
out = Path(sys.argv[2])
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0

probe = json.loads(
    subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
)["streams"][0]
W, H = int(probe["width"]), int(probe["height"])
num, den = probe["r_frame_rate"].split("/")
FPS = int(num) / int(den)


def opts(cls, model, **kw):
    base = mpt.BaseOptions(model_asset_path=str(MODELS / model))
    return cls.create_from_options(
        getattr(vision, cls.__name__ + "Options")(
            base_options=base, running_mode=vision.RunningMode.VIDEO, **kw
        )
    )


hand = opts(vision.HandLandmarker, "hand_landmarker.task", num_hands=2)
pose = opts(vision.PoseLandmarker, "pose_landmarker.task", num_poses=1)
face = opts(vision.FaceLandmarker, "face_landmarker.task", num_faces=1)

cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
size = W * H * 3


def pts(lms):
    return np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)


def assemble(hands, pose_pts, face_pts):  # noqa: PLR0912
    left = right = None
    if hands:
        sides = [h[0] for h in hands]  # fallback: MediaPipe label
        if pose_pts is not None:
            lw, rw = pose_pts[15][:2], pose_pts[16][:2]
            roots = [h[1][0][:2] for h in hands]
            if np.all(np.isfinite(lw)) and np.all(np.isfinite(rw)):
                if len(hands) == 1:
                    dl, dr = np.linalg.norm(roots[0] - lw), np.linalg.norm(roots[0] - rw)
                    if dl != dr:
                        sides = ["Left" if dl < dr else "Right"]
                elif len(hands) == 2:
                    st = np.linalg.norm(roots[0] - lw) + np.linalg.norm(roots[1] - rw)
                    cr = np.linalg.norm(roots[0] - rw) + np.linalg.norm(roots[1] - lw)
                    if st != cr:
                        sides = ["Left", "Right"] if st < cr else ["Right", "Left"]
        for side, (_, p) in zip(sides, hands, strict=True):
            if side == "Left" and left is None:
                left = p
            elif side == "Right" and right is None:
                right = p
    src = {"hand_left": left, "hand_right": right, "pose": pose_pts, "face": face_pts}
    frame = np.full((LAYOUT["n_landmarks"], 3), np.nan, np.float32)
    c = 0
    for g in LAYOUT["groups"]:
        s = src[g["source"]]
        for i in g["indices"]:
            if s is not None and i < len(s):
                frame[c] = s[i]
            c += 1
    return frame, (left is not None, right is not None, pose_pts is not None, face_pts is not None)


frames, present = [], []
i = 0
while True:
    buf = proc.stdout.read(size)
    if len(buf) < size or (limit and i >= limit):
        break
    img = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    ts = int(i * 1000 / FPS)
    hr = hand.detect_for_video(mpimg, ts)
    pr = pose.detect_for_video(mpimg, ts)
    fr = face.detect_for_video(mpimg, ts)
    hands = [
        (h[0].category_name, pts(lm))
        for h, lm in zip(hr.handedness, hr.hand_landmarks, strict=True)
    ]
    p = pts(pr.pose_landmarks[0]) if pr.pose_landmarks else None
    f = pts(fr.face_landmarks[0]) if fr.face_landmarks else None
    fm, pr_ = assemble(hands, p, f)
    frames.append(fm)
    present.append(pr_)
    i += 1
    if i % 500 == 0:
        print(f"{i} frames", flush=True)

out.mkdir(parents=True, exist_ok=True)
np.save(out / "landmarks.npy", np.stack(frames))
np.save(out / "present.npy", np.array(present, dtype=bool))
(out / "meta.json").write_text(
    json.dumps({"fps": FPS, "width": W, "height": H, "frames": i, "mediapipe": mp.__version__})
)
print("done", i, "frames", FPS, "fps")
