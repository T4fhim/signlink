# ADR-0004: Landmark layout `slk-landmarks-v1`

Status: accepted · 2026-10-04 · Phase 0 step 3
(ADR-0001 to 0003 cover landmarks-not-pixels, the gloss layer and the data tracks.)

## Context

Recognition runs on landmarks, not pixels (PLAN §4.1). The frame is `float32[T, N, 3]`, NaN = missing,
and `N` must be frozen before any normalization, model or dataset loader depends on it. Training data
(Kaggle ISLR) was extracted with legacy MediaPipe Holistic; the browser uses MediaPipe Tasks
(`@mediapipe/tasks-vision` 1.0.1) Hand, Pose and Face Landmarkers.

## Decision

**N = 146** landmarks, concatenated in this order. The machine-readable source of truth is
`packages/schemas/layouts/slk-landmarks-v1.json` (shape: `landmark_layout.v1.json`).

| Group | Source | Count | Source indices |
|---|---|---|---|
| `hand_left`, `hand_right` | hand | 2 × 21 | 0–20 (full hand topology) |
| `pose` | BlazePose (33) | 11 | nose 0, eyes 2 / 5, ears 7 / 8, shoulders 11 / 12, elbows 13 / 14, wrists 15 / 16 |
| `face_lips` | face mesh | 40 | `FaceLandmarker.FACE_LANDMARKS_LIPS` |
| `face_left_eyebrow`, `face_right_eyebrow` | face mesh | 2 × 10 | `FACE_LANDMARKS_*_EYEBROW` |
| `face_left_eye`, `face_right_eye` | face mesh | 2 × 16 | `FACE_LANDMARKS_*_EYE` (contours) |
| `face_nose_tip` | face mesh | 1 | 1 |

- The face indices are **derived from the library's own connection sets**, not typed from memory. A
  unit test fails if the committed layout drifts from them. Regenerate with
  `pnpm --filter @signlnk/landmarks write-layout`.
- Iris points (468–477) are excluded. Pose hands/fingers (BlazePose 17–22) are excluded because the
  hand landmarkers cover them.
- Values are MediaPipe's normalized image coordinates (x, y in [0, 1], z relative depth). Missing
  parts, out-of-range indices and non-finite values are NaN.
- Normalization (step 4) anchors on the shoulders: `anchors.left_shoulder = 47`,
  `anchors.right_shoulder = 48` (output-layout indices).
- Hand side: the Left/Right label MediaPipe Tasks returns for raw, unmirrored camera frames is used
  as the signer's own hand, with no swap. An initial version swapped the labels (MediaPipe documents
  its handedness for mirrored input) and was wrong: on 2026-10-04 raising the right hand reported
  "left" on `/dev/landmarks`. Measured once, on the benchmark laptop; `swapHandedness` stays as an
  option. Whether this matches Holistic's `left_hand` in the Kaggle ISLR data is checked in step 5
  **[VERIFY]**.

## Face index parity with Kaggle ISLR **[VERIFY]**

Hypothesis: the first 468 face-mesh indices of the Tasks Face Landmarker are the same points as in
legacy Holistic (Tasks adds 10 iris points at 468–477, which we exclude), so the face mapping is the
identity. This is **not yet verified**. `legacy_holistic_indices` is therefore deliberately absent
from the layout file until step 5 checks it against real ISLR parquet data and a recorded fixture.

## Runtime and measured cost

The landmarker runs in one Web Worker, three models in sequence per frame, GPU delegate with CPU
fallback. Pose and face can run every Nth frame, reusing the last result in between (PLAN §2.2
fallback). MediaPipe WASM and models are served from our own origin (`pnpm assets`, dev-time download,
SHA-256 pinned in `scripts/mediapipe-assets.json`; hashes were pinned on first download, so
trust-on-first-use).

Benchmark laptop: Intel Core i5-1135G7 (4 cores / 8 threads), 15.7 GB RAM, Intel Iris Xe Graphics,
Windows 11, Chrome 154, hardware WebGL2 (ANGLE / D3D11), Next.js dev server.

| Setting | Frames | fps | End to end p50 / p95 | Worker total p50 |
|---|---|---|---|---|
| pose + face every frame | 98 | 19.0 | 49.9 / 78.4 ms | 47.9 ms (hand 18.5, pose 18.9, face 9.8) |
| pose + face every 2nd frame | 588 | 34.0 | 27.7 / 49.9 ms | 22.4 ms (hand 15.4) |
| same, **real signer** | 1878 | **35.1** | 26.6 / 57.0 ms | 17.6 ms (p95 56.6; hand 11.4) |

The first two rows used a synthetic camera with no person in view (nothing detected). They showed
that running all three models every frame is too slow (19 fps). The third row is a real signer on
the benchmark laptop (report: `tests/landmarks/landmark-benchmark.json`, run in VS Code's embedded
Chromium 150) and **passes the ≥ 25 fps gate** with pose and face on every 2nd frame. The fps dips
briefly and recovers; worker p95 (56.6 ms) is inside the 60 ms per-frame landmarking budget (PLAN §2.2).

The benchmark page defaults to pose and face every 2nd frame. Held pose and face values are reused
on the skipped frames. If a later change (for example higher face-region accuracy) breaks the
budget, the next step is one worker per model (up to about 3× parallelism on 4 cores).

## Alternatives considered

- Full 478-point face mesh: more throughput cost and no extra non-manual signal beyond lips, brows
  and eyes.
- Hands only: loses mouthing and brow raise (non-manual markers), against PLAN §4.1.
- Pose world landmarks: the training data uses image-space coordinates.

## Open items

- **[VERIFY]** model licenses (MediaPipe model cards) before any model bundle ships.
- **[VERIFY]** left/right agreement with the Holistic-extracted training data, and face-index parity,
  both in step 5.
