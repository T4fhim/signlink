# Probe: landmarks from a YouTube tutorial video, scored by the release model

Date: 2026-10-24. Machine: home PC (AMD Ryzen 7 7840H). Track: **research** (ADR-0003). Status: probe
done, nothing trained, no vocabulary changed. Follows `idea-youtube-landmarks-and-nemotron.md`.
Tooling and how to rerun: `scripts/research/youtube_probe/README.md`.

## What was run

- Video: "25 ASL Signs You Need to Know | ASL Basics" (`0FcwzMq4iWg`, channel Learn How to Sign,
  394 s, uploaded 2021-03-23, 1280x720, 29.97 fps, 11,798 frames). `yt-dlp` reports `license=NA`, i.e.
  the standard YouTube licence, not Creative Commons. One signer. Captions are the video's own text
  with the gloss in capitals.
- Landmarks: MediaPipe Tasks 1.1.1 (Python) with the repo's three `.task` models, assembled into
  `slk-landmarks-v1` like `packages/landmarks/src/frame.ts` (hand side by the nearer pose wrist,
  ADR-0008), then `to_reference_aspect` and `normalize` from `signlnk_ml`. Hands in view in 89.5% of frames.
- Scoring: the staged release ONNX model (`apps/web/public/model`, source commit `e72ff37`, 250 signs +
  `<other>`). Per sign, a segment runs from its first caption time to the next sign's. Sliding windows
  (24, 36, 48, 72 frames, stride 6, hands in at least half the frames) go through `make_input` /
  `resample_nearest`; the prediction is the class with the highest peak softmax over the segment.
- Command: `SIGNLNK_DATA_DIR=D:\signlnk-data uv run --no-sync python scripts/research/youtube_probe/probe_25.py`
  at repo commit `1e26e97`. Raw output: `$SIGNLNK_DATA_DIR/research-videos/0FcwzMq4iWg/probe.json`.

## Result

16 of the 25 signs map to a model class (my reading of the gloss, **[VERIFY]** by a fluent signer): hello,
dad (FATHER), mom (MOTHER), yes, no, please, thankyou, dog, cat, food (EAT FOOD), milk, go (GO-TO), potty
(BATHROOM), fine, like, finish. The other 9 have no class (SEE YOU LATER, ME, HELP, WANT, WHAT, AGAIN, MORE,
LEARN, SIGN) and are not scored.

| Scoring over | top-1 | top-3 | top-5 | chance top-1 |
|---|---|---|---|---|
| all 250 classes | 7/16 | not computed | 12/16 | 0.4% |
| only the 16 mapped classes | 14/16 | 15/16 | not computed | 6.3% |

Rank of the true class (250 classes / 16 classes): hello 1/1, dad 1/1, mom 2/1, yes 2/1, no 19/4, please 1/1,
thankyou 1/1, dog 10/1, cat 2/1, food 2/1, milk 1/1, go 9/1, potty 14/2, fine 1/1, like 3/1, finish 1/1.

## Why this is optimistic, and not an accuracy number

1. Segment boundaries come from the label's own caption times; a real decoder has no such hint.
2. The prediction is a peak over 70 to 290 windows per sign, which favours any class that fires once.
3. Segments include the teacher's explanations, not only the citation sign.
4. 16 signs, one signer, my own gloss-to-class mapping. The masked 16-way task is far easier than the
   20 to 50 sign list the product will use.
5. Nothing here is a signer-independent held-out split; it is a domain-compatibility check only.

## Verdict: how good an alternative is this?

- **Replacement for fluent-signer data: no.** One video is one signer (Kaggle has 21; the baseline still
  overfits its 15 training signers, train 0.85 to 0.99 vs test 0.46 to 0.48). Labels from captions are
  unverified, tutorial signing is slow and exaggerated, and the signer's fluency is `[VERIFY]`.
- **Compatible input domain: yes, with evidence.** A model trained only on Kaggle landmarks reads
  landmarks from a different camera, resolution, extractor build and signer well enough to rank the true
  sign first or second for 11 of 16 signs (all 250 classes). That removes the biggest unknown behind
  "learn from videos".
- **Useful as a research-track supplement and a free test source: probably, unproven.** The value would
  come from many signers, not many clips of one.
- **Cannot reach a release model** unless the videos are Creative Commons or the creator grants written
  permission (ADR-0003). Crediting the creator is not a licence. Release-track diversity needs Studio
  recordings under the consent form.
- **Humans are not removed.** Deaf advisors still choose and sign off the vocabulary (G0) and review
  anything Deaf users see (PLAN §9). You, as a held-out tester, are still the check against real camera
  data, and your recordings must never enter training.

## Can we keep going? Yes, as a staged research-track experiment

Each stage has a stop condition so it does not grow into a scraping project before G0.

1. **Fix the evaluation (no training).** Segment by motion endpoints instead of label time ranges; report
   unmasked and masked scores; add 2 to 3 more videos with different signers, preferring Creative Commons
   (`yt-dlp --print license`). Stop if the unmasked top-5 collapses on new signers.
2. **Train with it (research track).** Add video-derived clips to the Kaggle training set; evaluate on the
   held-out Kaggle test signers and on your own recordings. Worth continuing only if the gain on signer
   29302 and on test top-1 exceeds seed noise (about 1.4 points on test, `phase1-step4-seeds.md`).
3. **Release path.** Only Creative Commons or permissioned sources, or Studio recordings; fluent-signer
   review of labels; then a `/dataset-license` entry per source.

## Licence and data handling

- `docs/datasets.md` and `ml/datasets.yaml` are **not edited** here (their edits need your approval).
  Pending entry: source URL above, track `research`, licence "standard YouTube licence (`license=NA`)",
  downloaded 2026-10-24 to `D:\signlnk-data\research-videos\0FcwzMq4iWg`, not redistributable.
- Video and landmarks are outside the repo; landmarks are personal data and are never committed.
- `mediapipe` 1.1.1 (Python) is used only in a throwaway venv and is not in `pyproject.toml`.
  `license-auditor` (2026-10-24, from the installed package metadata): mediapipe is Apache-2.0 and the
  other 19 packages it pulled in are permissive (certifi is MPL-2.0; opencv-contrib-python bundles
  LGPL-2.1 FFmpeg, which matters only if redistributed). No API key. **Telemetry flag (rule 2):** its
  `NOTICE` says MediaPipe Tasks "send metrics about the performance and utilization of the APIs ... to
  Google" (input images and video stay on device), and `libmediapipe.dll` contains a logging client and the
  string `https://play.googleapis.com/log`. Whether it sent anything during the probe was **not observed**
  **[VERIFY]**. Do not add it as a dependency, and run it with outbound network blocked until this is
  checked. The same question applies to `@mediapipe/tasks-vision` in the web app **[VERIFY]**
  (`privacy-reviewer`).
