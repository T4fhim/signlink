# YouTube landmark probe (research track)

Dev-time experiment, not part of the product. Question: can landmarks extracted from a tutorial
video feed or test the recognizer without recruiting signers? Result and verdict:
`docs/reports/phase1-youtube-probe.md`.

Everything it produces is **research-track** (ADR-0003): the video has the standard YouTube licence,
so no output may reach a release model or bundle. Videos and landmarks stay in `$SIGNLNK_DATA_DIR`
(`research-videos/<video-id>/`), never in git; landmarks are personal data.

## Run

1. Download (yt-dlp, ffmpeg on PATH):
   `yt-dlp -f "bv*[height<=720]+ba/b[height<=720]" --merge-output-format mp4 --write-subs --write-auto-subs --sub-langs en --sub-format json3 -o "video.%(ext)s" <url>`
   into `$SIGNLNK_DATA_DIR/research-videos/0FcwzMq4iWg/`.
2. Extract landmarks in a throwaway venv (not in `pyproject.toml`; mediapipe is not a project
   dependency): `uv venv venv && uv pip install --python venv/Scripts/python.exe mediapipe numpy`, then
   `venv/Scripts/python.exe scripts/research/youtube_probe/extract_landmarks.py <video.mp4> <out-dir>`.
   Writes `landmarks.npy` (`[T, 146, 3]`, `slk-landmarks-v1`, raw, NaN = missing), `present.npy`,
   `meta.json`. Uses the repo's `.task` models (`pnpm assets`) and mirrors `frame.ts`
   (hand side by the nearer pose wrist, ADR-0008).
3. Score the release model: `SIGNLNK_DATA_DIR=... uv run --no-sync python
   scripts/research/youtube_probe/probe_25.py` (needs the staged model in `apps/web/public/model/`).
   Writes `probe.json` next to the run folder.

Dev-time only: mediapipe 1.1.1 (Apache-2.0) stays in the separate venv and is not a project
dependency. Its `NOTICE` says it sends usage metrics to Google (not images or video); run it with
outbound network blocked until that is verified (report, "Licence and data handling").

Stage 1 (label-free segmentation with the decoder's endpointer, several videos; edit `VIDEOS` for
more): `SIGNLNK_DATA_DIR=... uv run --no-sync python scripts/research/youtube_probe/probe_stage1.py`.
It expects each video's `full/` folder from step 2 and its caption file; writes `probe_stage1.json`.

Fine-tune experiment (leave-one-video-out, CPU, research track; needs the Kaggle sample cache
`python -m signlnk_ml.training.cache --sample --out-dir $SIGNLNK_DATA_DIR/kaggle-islr/sample-cache`):
`SIGNLNK_DATA_DIR=... uv run --no-sync python scripts/research/youtube_probe/finetune_loo.py`, config
`ml/configs/research-youtube-finetune.yaml`. Writes `finetune_loo.json`; saves no model.

`probe_25.py` hard-codes this one video's caption times and the gloss-to-class mapping (`[VERIFY]`
by a fluent signer). For another video, edit `SIGNS` and `END`.
