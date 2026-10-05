# ADR-0006: Kaggle ISLR → slk-landmarks-v1, and train/serve parity

Status: accepted · 2026-10-04 · Phase 0 step 5
Builds on ADR-0004 (layout) and ADR-0005 (normalization). Dataset record: `docs/datasets.md`.

## Context

Training data (Kaggle ISLR, 21 signers, 250 signs) was extracted with legacy MediaPipe Holistic. The
browser uses MediaPipe Tasks. Any difference in landmark numbering, left/right convention, mirroring
or coordinate scale becomes a silent accuracy loss at serving time.

## Decision

**Loader.** `ml/signlnk_ml/data/kaggle_islr.py` turns a long-form parquet (columns `frame`,
`row_id`, `type`, `landmark_index`, `x`, `y`, `z`; 543 rows per frame: face 468, left_hand 21,
pose 33, right_hand 21) into the dense legacy tensor `[T, 543, 3]`, then picks the layout's 146
landmarks. The mapping lives in the layout file as `legacy_holistic_indices` (add-only field of
`landmark_layout.v1`), written by `write-layout`. It is the **identity**: Holistic and Tasks share
hand, BlazePose and face-mesh numbering for the chosen landmarks.

**Sample, not the full set.** The dataset is about 40 GB and C: has little room, so
`fetch_islr` downloads `train.csv`, the sign map and a participant-balanced sample (25 sequences per
participant spread over signs): **525 sequences, 20,161 frames, 311 MB**, in
`D:\signlnk-data\kaggle-islr` (`SIGNLNK_DATA_DIR`). Real-data tests skip when it is absent (CI).

## Evidence on the Kaggle side (525 sequences)

`geometry_checks` asserts anatomy that any correct, unmirrored recording satisfies. Mean fraction of
frames that agree, over sequences with the parts present:

| Check | Agreement |
|---|---|
| `left_hand` wrist nearer the pose left wrist than the right | 0.991 (220 sequences) |
| `right_hand` wrist nearer the pose right wrist | 0.987 (323) |
| lips below the nose tip | 1.000 |
| each brow above its eye | 1.000 |
| left / right eye group beside BlazePose's left / right eye | 0.998 / 0.997 |
| subject's left eye on the image's right (unmirrored) | 1.000 |

So: Kaggle's `left_hand` is the subject's left hand; the images are **unmirrored**; and the layout's
face groups (derived from MediaPipe's own connection sets) sit where their names say. A few
sequences have a hand mislabelled by Holistic (per-sequence agreement 0), which is label noise in
the data, not a mapping error. This matches the browser pipeline: on 2026-10-04 raising the right
hand reported "right" on raw frames (ADR-0004).

## Parity statistics and limits

`landmark_stats` normalizes each sequence (ADR-0005) and takes per-landmark mean, std and presence;
`compare` reports the worst per-landmark difference over a landmark subset. Baseline: leave one of
the 21 participants out, compare them with the other 20 (worst of the 21 | median of the 21):

| Subset | xy mean | xy std | z mean | z std | presence |
|---|---|---|---|---|---|
| face (93) | 0.128 \| 0.060 | 0.051 \| 0.033 | 0.046 \| 0.019 | 0.015 \| 0.010 | 0.109 \| 0.010 |
| head: nose, eyes, ears (5) | 0.116 \| 0.052 | 0.048 \| 0.025 | 0.719 \| 0.199 | 0.309 \| 0.141 | 0 \| 0 |
| arms, hands | not usable (see below) | | | | |

Limits in `ml/tests/test_islr_real_data.py` (about 1.5–2× the worst participant): face xy mean 0.20,
xy std 0.10, z mean 0.10, z std 0.04, presence 0.25; head xy mean 0.20, xy std 0.10, presence 0.05.
Pose z is not compared (it depends on the camera: worst z mean 0.72). Units are shoulder widths.

Hands and arms are **not** distribution-compared: 17 of the 21 participants sign with one hand
(6 left, 11 right; 4 mix), so a hand landmark's mean and presence differ far more between people
(worst xy mean 0.41, presence 0.52) than between pipelines. They are covered by the geometry checks.
This also means left-dominant signers are common (6 of 21): mirror augmentation matters in Phase 1.

Controls (real data): scrambling the face indices moves the face xy mean above the limit, and swapping
the hand groups drops hand agreement below 0.2, so the tests can fail when they should.

## What is not done yet

- **Serve side.** The train/serve test is wired to `$SIGNLNK_DATA_DIR/serve-recordings/*.npy` and
  skips until a browser recording exists; the `/dev/record` page (step 6) produces them. Until then
  the Tasks-side face-index and hand-side semantics are only supported by the library's own
  connection sets and the hand test in ADR-0004. **[VERIFY]**: run geometry checks and the face and
  head parity limits on a real recording.
- Recordings hold face landmarks, so they are personal data: local only, gitignored, never committed.
- Aspect-ratio and Holistic-vs-Tasks z differences, if any, would show up in the same comparison.
