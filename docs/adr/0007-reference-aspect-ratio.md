# ADR-0007: Coordinates in a reference aspect ratio

Status: accepted · 2026-10-05 · Phase 0 steps 5–6 (found by the first real browser recordings)
Resolves the aspect-ratio limitation in ADR-0005. Builds on ADR-0004 and ADR-0006.

## Context

The first recording of a real signer (4:3 webcam, 102 frames) passed all 8 geometry checks at 1.000,
so landmark indices, hand sides and orientation match the Kaggle data. It **failed the train/serve
parity test**: worst face landmark mean offset 0.507 and head 0.465 shoulder widths, against limits
of 0.20 and a worst-of-21 Kaggle baseline of 0.128. Spread (std) and z were fine.

MediaPipe returns x in units of image width and y in units of image height. On a 4:3 landscape
webcam and on the portrait phone video Kaggle ISLR was recorded on, the same face has different
vertical proportions in these units. Horizontal proportions agree; vertical ones do not
(Kaggle median over 21 signers [min–max] vs that first recording):

| Proportion | Kaggle | Recording |
|---|---|---|
| eye distance / shoulder width (horizontal) | 0.234 [0.212–0.276] | 0.213 |
| nose above neck / shoulder width (vertical) | 0.401 [0.306–0.469] | 0.787 |
| eyes-to-lips / eye distance (vertical vs horizontal) | 0.800 [0.552–0.918] | 1.515 |

The vertical proportions differ by about 1.9–2.0×.

## Decision

Every pipeline expresses landmarks in one **reference aspect ratio** (width / height),
`reference_aspect = 0.78`, stored in `slk-landmarks-v1.json` (add-only field of `landmark_layout.v1`).
A frame from an image of width W and height H is converted with

`y' = y · reference_aspect / (W / H)`   (x and z unchanged, NaN stays NaN)

- Kaggle ISLR is the reference by definition and is left as loaded. 0.78 is an *effective* value:
  the parquet files carry no image size.
- The browser worker applies the conversion to every emitted frame, using the video frame's real
  size (`bitmap.width / height`), so recordings, the decoder and Studio all see training-like y.
- `aspect.ts` and `ml/signlnk_ml/features/aspect.py` mirror each other; `tests/fixtures/aspect/cases.json`
  (four image sizes, NaN dropout) is checked by both within 1e-5, plus hand-computed numbers
  (640×480 → 0.585, square → 0.78, the reference aspect → 1).
- Any other source (Studio recordings, other datasets) must be converted with its own W and H before
  its landmarks are mixed with Kaggle's.

## Evidence for the value

**First attempt (one recording): 0.70.** Scaling that recording's y and re-running parity gave a
flat minimum for y scale 0.50–0.55, i.e. `0.525 = 0.70 / (4/3)`. One person is too little: vertical
face proportions vary by about ±25% between Kaggle signers.

**Calibration on 7 recordings (several people, one 4:3 webcam): 0.78.** Sweeping the reference aspect
A over the 7 distinct recordings (one duplicate excluded; whether the no-face clip was in the sweep
is `[VERIFY]`: 6 clips remain usable once it is excluded), worst and mean
over clips of the face and head xy mean offset (limit 0.20, Kaggle worst-of-21 baseline 0.128 / 0.116):

| A | 0.66 | 0.70 | 0.74 | 0.76 | **0.78** | 0.80 | 0.84 | 0.90 |
|---|---|---|---|---|---|---|---|---|
| face worst | 0.171 | 0.146 | 0.122 | 0.111 | **0.111** | 0.111 | 0.113 | 0.162 |
| face mean | 0.111 | 0.096 | 0.085 | 0.081 | **0.078** | 0.077 | 0.084 | 0.107 |
| head worst | 0.132 | 0.111 | 0.090 | 0.081 | **0.081** | 0.090 | 0.114 | 0.157 |
| head mean | 0.080 | 0.066 | 0.062 | 0.065 | **0.067** | 0.070 | 0.076 | 0.090 |

The minimum of mean(face) + mean(head) is at 0.78, with a flat region from 0.76 to 0.80. Individual
clips prefer anything from 0.66 to 0.86, which is ordinary person-to-person variation. At 0.78 the
worst clip is inside the Kaggle between-person baseline. Face xy std (0.062–0.075) and z are inside
their limits for every clip.

## Limits and follow-ups

- **[VERIFY]** All recordings come from one webcam. A different camera (another lens, a 16:9 sensor)
  may prefer a different effective value; test with one when available. The conversion itself uses
  each frame's real size, so only the constant could move: change `REFERENCE_ASPECT` in
  `layoutSpec.ts`, run `write-layout` and `golden_fixtures`, and update the numbers in the aspect tests.
- Vertical proportions vary about ±25% between Kaggle signers (eyes-to-lips / eye distance
  0.55–0.92), so Phase 1 training augmentation should include y-scale jitter.
- The recordings in `serve-recordings/` were rescaled from the 0.70 version by 0.78 / 0.70 when the
  constant changed; the 0.70 originals are kept in `serve-recordings/a0.70-backup/`, and the very first
  raw recording in `serve-recordings/pre-aspect-fix/`. The parity test does not read subfolders.
