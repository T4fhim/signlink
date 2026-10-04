# ADR-0005: Landmark normalization and TS ↔ Python parity

Status: accepted · 2026-10-04 · Phase 0 step 4
Builds on ADR-0004 (layout `slk-landmarks-v1`, shoulder anchors at indices 47 and 48).

## Context

Training (`ml/signlnk_ml/features`) and the browser (`packages/landmarks`) must feed the model
identical numbers (CLAUDE.md rule 6). Any divergence silently degrades accuracy at serving time.

## Decision

Per frame, in this order:

1. `center = (left_shoulder + right_shoulder) / 2` in x and y (the neck).
2. `width = sqrt(dx² + dy²)` between the shoulders in x and y.
3. `x' = (x - center_x) / width`, `y' = (y - center_y) / width`. **z is kept unchanged** (PLAN §4.1).
4. A frame is **invalid** when either shoulder x or y is missing/non-finite, or `width < 1e-6`
   (`MIN_SHOULDER_WIDTH`). An invalid frame becomes all NaN. Other missing landmarks stay NaN.

Mirroring for left-dominant signers is a training augmentation, not part of this transform.

Both implementations compute in float64, in the same operation order, and round to float32 once on
store: `ml/signlnk_ml/features/normalize.py` (reference) and
`packages/landmarks/src/normalize.ts` (`normalizeFrame`, `normalizeSequence`).

## Parity contract

- Golden fixtures live in `tests/fixtures/normalization/*.json` (`hand_computed`, `random_clip`,
  `edge_cases`; shape `[T, 146, 3]`, `null` = NaN). `uv run python -m
  signlnk_ml.features.golden_fixtures` regenerates them from the Python reference.
- `pnpm test` (vitest) and `uv run pytest` both check the same files: NaN positions must match
  exactly and every other value within **1e-5** max abs diff.
- Hand-computed numbers and invariances (translation, uniform scale, z kept, shoulders one unit apart)
  check the maths independently of the generated expectations.
- Changing the transform in one language fails the other language's tests until both are changed and
  the fixtures regenerated. Checked on 2026-10-04 by nudging the scale in each implementation: 5 TS
  and 5 Python tests failed, and passed again after restoring.

## Known limitations

- x and y are MediaPipe's normalized image coordinates, so on a non-square frame (for example 4:3)
  the x and y units differ and `width` mixes them. Kaggle ISLR is believed to use the same image-space
  coordinates with unknown aspect ratios **[VERIFY]** in step 5, so no correction is applied. Revisit
  if step 5 shows a distribution gap.
- z is not centred or scaled. MediaPipe's z has a different origin per source (hand, pose, face),
  so the three groups' z values are not mutually comparable. Kept as the plan specifies.
- Normalization is per frame. A frame without both shoulders is lost even if its neighbours are fine;
  with pose run every 2nd frame the last pose is reused, so this should be rare. Smoothing or
  carrying the last valid shoulders forward is a decoder decision for Phase 1.
- `MIN_SHOULDER_WIDTH = 1e-6` is a guard against dividing by about zero, not a tuned value.

## Not in scope here

The live pipeline still emits raw landmarks; the decoder applies `normalizeSequence` in Phase 1.
Distribution parity against the Kaggle ISLR data is step 5.
