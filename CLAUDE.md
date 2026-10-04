# CLAUDE.md — SignLnk

Read `PROJECT_CONTEXT.md` and `docs/PLAN.md` before any task. Work only on the current phase.

## Hard rules
1. $0 software cost. Only free/OSS dependencies. Never add a paid SaaS SDK or anything that needs an API key.
2. Raw video and audio never leave the device. No telemetry or third-party scripts.
3. ASL (`ase`) ↔ English (`en`) only, but never hard-code the language: read it from config.
4. Data licensing: every training config declares `track: release|research`. Never let research-track data
   into a release model or bundle. Never copy ASL-LEX / Signbank descriptive data into `lexicon/core/`.
5. Interfaces in `packages/schemas/*.json` are the source of truth. Add fields only. Regenerate TS + Pydantic
   types after any schema change; CI must fail if they are stale.
6. Normalization lives in two places (`packages/landmarks`, `ml/signlnk_ml/features`). Any change to one must
   change the other and pass the golden-fixture parity test.
7. Do not invent benchmark numbers, dataset sizes or licenses. Mark unknowns `[VERIFY]`.

## Conventions
- Python 3.11+, type hints everywhere, Ruff + mypy (strict on `ml/` and `services/`), pytest.
- TypeScript strict, ESLint + Prettier, Vitest, Playwright for e2e.
- Small modules, one responsibility each. Pin exact dependency versions (lockfiles committed);
  check current stable versions at install time rather than assuming.
- Package managers: pnpm (JS), uv (Python).
- Commits: Conventional Commits. One logical change per commit.

## Commands (keep this section updated as you create them)
- `pnpm install && uv sync` — install
- `pnpm dev` — web app
- `pnpm test` / `uv run pytest` — tests
- `pnpm gen:types` — regenerate types from schemas
- `docker compose up` — API + Postgres

## Toolkit
Agents, skills, MCPs and their phase status: `docs/TOOLKIT.md`. Use only ACTIVE items for the current phase.

## End of every session
Print a block titled `Log to context:` with one line per decision or change, for PROJECT_CONTEXT.md.

---

## First task: Phase 0 (PLAN §7, Phase 0 steps 1–6)
1. Scaffold the monorepo exactly as in PLAN §3, with CI (lint, type-check, tests, schema-gen check).
2. Write the five JSON Schemas from PLAN §4 and the type-generation pipeline.
3. Build `packages/landmarks`: MediaPipe Tasks hand+pose+face in a Web Worker, producing `slk-landmarks-v1`.
   Propose the exact landmark index subset in `landmark_layout.v1.json` and explain it in an ADR.
4. Implement normalization in TS and Python, with golden fixtures in `tests/fixtures/` and a parity test.
5. Build a Kaggle ISLR loader that maps the legacy Holistic layout to `slk-landmarks-v1`, plus the
   train/serve distribution parity test. Investigate and document the face-index mapping `[VERIFY]`.
6. Build the `/dev/record` page with an fps counter, saving sequences the Python loader can read.

Stop after each step, run the tests, and summarize before continuing.
