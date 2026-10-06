# ADR-0001: Recognition runs on landmarks, not pixels

Status: accepted · 2026-10-06 · Phase 0 step 7
Underpins ADR-0004 (layout), ADR-0005 (normalization) and ADR-0007/0008 (serve-side parity).

## Context

The product must work on-device, offline, on a laptop with no discrete GPU (benchmark laptop: i5-1135G7,
Iris Xe), and raw video must never leave the device (CLAUDE.md rule 2). Training uses free Kaggle/Colab
tiers, and the main release-track dataset (Kaggle ISLR) is distributed as landmarks, not video.

Measured in Phase 0 step 3 (PR #3): MediaPipe hand + pose + face in a Web Worker runs at 35.1 fps, worker
p95 56.6 ms, on the benchmark laptop (target ≥ 25 fps, PLAN §2.2 budget ≤ 60 ms/frame).

## Decision

The recognizer's input is a landmark sequence `float32[T, N, 3]` in layout `slk-landmarks-v1`, produced
in the browser by MediaPipe Tasks. No model in the project consumes pixels. Pixels never leave the capture and
landmark path (camera preview, then the landmark worker, which closes each frame bitmap after use).

## Consequences

- Raw video stays on device by construction; the network never needs an image or audio frame.
- Training data from different cameras and people is comparable after normalization, which makes
  train/serve parity (ADR-0005/0007/0008) the main source of risk, and a tested one.
- Landmarks, especially face points, can identify a person: they are personal data (PLAN §6.5), so
  recordings stay out of the repo and are downloaded only on explicit user action.
- Loses what landmarks do not capture (fine finger contact, some facial grammar, lighting-dependent cues);
  facial-grammar coverage stays an open risk (PROJECT_CONTEXT).
- Fixes the model family to sequence models over landmarks (PLAN §7 Phase 1 step 3).

## Verification

- Worker benchmark: `/dev/landmarks`, results in `tests/landmarks/landmark-benchmark*.json`.
- Parity tests: TS↔Python golden fixtures (≤ 1e-5) and the real-data train/serve test
  (`ml/tests/test_islr_real_data.py`).
- Privacy check: `privacy-reviewer` on PR #6 found no network calls, no storage, camera `audio: false`.
