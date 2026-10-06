# CLAUDE.md — SignLnk

Free, open-source, on-device bridge between ASL signers and English speakers. Current state,
decisions and interfaces: @PROJECT_CONTEXT.md

## At the start of every task
1. Check the task against **Current phase** (in PROJECT_CONTEXT above). Later-phase work: flag it
   in one line and offer only the minimum needed now.
2. Find the task's row in "Read and use when" below and follow it before acting.
3. Read only the section of `docs/PLAN.md` the task needs; don't load the whole file.

## Read and use when
| Situation | Read / run first |
|---|---|
| Working a PLAN step | `/step <phase.step>`; PLAN §7 step text and the ADRs it names |
| Landmarks, normalization, loaders, fixtures, `/dev` pages | `.claude/rules/landmarks.md` (auto); ADR-0004…0008 |
| Schemas or generated types | `/schema-change`; `.claude/rules/schemas.md` (auto); PLAN §4 |
| Dataset, pretrained model, lexicon source | `/dataset-license`; `docs/datasets.md`; PLAN §5 |
| ML code or training configs | `.claude/rules/ml.md` (auto); PLAN §7–8 |
| Web app, capture, assets, networking | `.claude/rules/web-privacy.md` (auto); `apps/web/AGENTS.md`; then `privacy-reviewer` |
| Lexicon, consent, review sign-offs | `.claude/rules/lexicon-data.md` (auto); PLAN §6, §9–10 |
| New lasting decision | `/adr` |
| Before any commit | `/verify` |
| Step finished | `plan-reviewer`; plus `privacy-reviewer` / `license-auditor` if their areas changed |
| New dependency | `license-auditor` (rule 1) |
| How to work (sessions, context, parallel work, permissions) | `docs/WORKFLOW.md` |
| Which agent, skill or MCP to use | `docs/TOOLKIT.md` |
| Something shown to Deaf users | Community gate, PLAN §9 |
| End of session / end of phase | `/session-close` / `/phase-gate <n>` |

"(auto)" rules load by themselves when you read or edit matching files. The prompt-router hook also
adds a one-line pointer to the right row when a prompt matches it; follow it.

## Hard rules
1. $0 software cost. Only free/OSS dependencies. Never add a paid SaaS SDK or anything that needs an API key.
2. IMPORTANT: raw video and audio never leave the device. No telemetry, analytics or third-party scripts.
3. ASL (`ase`) ↔ English (`en`) only, but never hard-code the language: read it from config.
4. Data licensing: every training config declares `track: release|research`. Never let research-track data
   into a release model or bundle. Never copy ASL-LEX / Signbank descriptive data into `lexicon/core/`.
5. Interfaces in `packages/schemas/*.json` are the source of truth. Add fields only. Regenerate TS + Pydantic
   types after any schema change; CI fails if they are stale.
6. Normalization lives in two places (`packages/landmarks`, `ml/signlnk_ml/features`). Any change to one must
   change the other and pass the golden-fixture parity test.
7. Do not invent benchmark numbers, dataset sizes or licenses. Mark unknowns `[VERIFY]`.

Hooks in `.claude/hooks/` enforce parts of rules 2, 4, 5 and 6 (docs/WORKFLOW.md §8). When a hook
blocks, reminds or routes you, follow its message; never work around it.

## Conventions
- Python 3.11+, type hints everywhere, Ruff + mypy strict (`ml/`, `services/`), pytest.
- TypeScript strict, ESLint + Prettier, Vitest, Playwright for e2e. Markdown is formatted by hand.
- Pin exact dependency versions (lockfiles committed); check current stable versions at install time.
- pnpm (JS), uv (Python). Never edit `pnpm-lock.yaml`, `uv.lock` or generated types by hand.
- Branches `feat/phase<P>-<slug>` from `main`; PR to `main`; Conventional Commits, one logical change each.
- Decisions with lasting effect get an ADR in `docs/adr/` (`/adr`).

## Commands (keep this section updated as you create them)
- `pnpm install && uv sync` — install (on Windows; the Cowork VM can't reach the npm registry)
- `pnpm dev` — web app (fetches MediaPipe WASM + models first; `pnpm assets` does just that)
- `pnpm test` / `uv run pytest` — tests
- `pnpm lint` / `pnpm typecheck` / `uv run ruff check . && uv run mypy ml services` — static checks
- `/verify` — the full CI-equivalent suite via the test-runner subagent
- `pnpm gen:types` — regenerate types from schemas (`pnpm gen:types:check` fails if stale)
- `pnpm --filter @signlnk/landmarks write-layout` — regenerate `slk-landmarks-v1` from MediaPipe
- `uv run python -m signlnk_ml.features.golden_fixtures` — regenerate normalization golden fixtures
- `pnpm --filter @signlnk/landmarks write-npy-fixture` — regenerate the golden `.npy` shared by the TS writer and the Python loader
- `/dev/record` (after `pnpm dev`) — record 3 s of landmarks and download an `.npy`; save it in `$SIGNLNK_DATA_DIR/serve-recordings/`
- `/dev/landmarks` — fps benchmark page
- `SIGNLNK_DATA_DIR=D:\signlnk-data uv run python -m signlnk_ml.data.fetch_islr` — fetch the Kaggle ISLR sample (needs `uv tool install kaggle`, a token in `~/.kaggle/access_token`, rules accepted)
- `SIGNLNK_DATA_DIR=D:\signlnk-data uv run pytest` — also runs the real-data train/serve parity tests (they skip without data)
- `docker compose up` — Postgres (API service arrives in Phase 1)

## Gotchas
- Windows host, Git Bash for hooks/skills. `.gitattributes` forces LF; CRLF warnings on generated
  files are expected.
- Real data lives in `$SIGNLNK_DATA_DIR` (`D:\signlnk-data`), never in the repo. Set it in
  `.claude/settings.local.json` `env`, not in committed settings.
- `apps/web` uses a Next.js version newer than your training data: read `apps/web/AGENTS.md` and the
  bundled docs before touching it.

## Workflow
- One PLAN step per session: `/step <phase.step>`. Finish with `/session-close`.
- Verify with evidence (commands + output, measured numbers). Before calling a step done, run `/verify`
  and the `plan-reviewer` subagent; add `privacy-reviewer` / `license-auditor` when their areas change.
- Use subagents for wide searches and long test runs. `/clear` between unrelated tasks.
- Anything Deaf users will see needs its community gate (PLAN §9) before it is called correct.
- When compacting, always keep: the current step and its Done-when, files modified, commands run with
  results, open failures, and any `[VERIFY]` items.

## End of every session
Run `/session-close`. It prints a `Log to context:` block (one line per decision or change) and
updates PROJECT_CONTEXT.md, docs/CHANGELOG.md and the Commands above.
