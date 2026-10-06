---
paths:
  - "packages/landmarks/**"
  - "ml/signlnk_ml/features/**"
  - "ml/signlnk_ml/data/**"
  - "packages/schemas/layouts/**"
  - "tests/fixtures/**"
  - "apps/web/app/dev/**"
---
<!-- Loads when Claude reads or edits landmark, normalization, loader, fixture or /dev page files. -->
# Landmark pipeline rules (`slk-landmarks-v1`)

- Read the ADR before changing behaviour it governs: layout and index subset ADR-0004;
  normalization ADR-0005; Kaggle Holistic mapping and parity ADR-0006; `reference_aspect` ADR-0007;
  hand assignment by nearer pose wrist ADR-0008.
- Normalization exists twice: `packages/landmarks/src/{normalize,aspect}.ts` and
  `ml/signlnk_ml/features/{normalize,aspect}.py`. Change both in the same commit and run
  `pnpm --filter @signlnk/landmarks test` and
  `uv run pytest ml/tests/test_normalize.py ml/tests/test_aspect.py ml/tests/test_geometry_parity.py`.
- Golden fixtures are the contract. Regenerate them (`uv run python -m signlnk_ml.features.golden_fixtures`,
  `pnpm --filter @signlnk/landmarks write-npy-fixture`) only for an intended behaviour change, and say so.
- Layout changes go through `layoutSpec.ts` + `pnpm --filter @signlnk/landmarks write-layout`; the
  layout is add-only (new fields), indices are frozen for v1.
- Missing landmark = NaN; an invalid frame is all NaN. Never replace NaN with 0.
- Real recordings and the Kaggle sample live in `$SIGNLNK_DATA_DIR`; real-data tests skip without it.
  Landmarks (face points especially) are personal data: never commit them, never send them anywhere.
- Performance budget (PLAN §2.2): ≥25 fps on the benchmark laptop, worker within budget; measure on
  `/dev/landmarks` before and after any worker change and report both numbers.
