# ADR-0003: Two data tracks (release and research)

Status: accepted · 2026-10-06 · Phase 0 step 7
Relates to PLAN §5, CLAUDE.md rule 4, `docs/datasets.md`.

## Context

Commercial use is "not now, door open". Useful datasets are non-commercial (ASL Citizen, WLASL,
How2Sign; Sem-Lex unverified, assumed non-commercial), while the Kaggle ISLR data permits any purpose
under its rules plus CC BY 4.0 (verified 2026-10-04, `docs/datasets.md`). Mixing them silently would make
a model unusable commercially later, and that is hard to undo.

## Decision

Every dataset, training config and model is on one of two tracks:

- **release**: permissive licences only (Kaggle ISLR, own Studio recordings under CC BY 4.0). Only this
  track may feed a model or bundle that ships.
- **research**: non-commercial data allowed, for benchmarking and experiments. Never shipped.

Rules:
- Each `ml/configs/*.yaml` declares `track: release|research`.
- A release model or bundle is refused if any research-track data was used.
- `lexicon/core/` holds SignLnk CC BY 4.0 content only. ASL-LEX (CC BY-NC) lives in an optional
  `lexicon/reference/` pack, never bundled; Signbank is used for naming conventions and links only.
- Licence, version and download date of each dataset go in `docs/datasets.md`; unknowns are `[VERIFY]`.

## Consequences

- Research experiments stay cheap, and the release path stays clean.
- Enforcement is partial. Two hooks exist: `stop-gate.mjs` blocks a training config that has no
  `track:` line, and `guard-paths.mjs` guards writes to `lexicon/core/`. Not built yet, because
  `ml/configs/` has no configs: the CI check that a release config uses only release-track datasets,
  and the export refusal (PLAN §5.1). Build both with the first training config (Phase 1 step 3).
- Re-check the Kaggle ISLR Rules page before the first public release (carried `[VERIFY]`).

## Verification

- Today: `docs/datasets.md` lists track per dataset; `license-auditor` on PR #6 found no research data
  in any release path.
- Phase 1: a test that fails when a release config references a research dataset.
