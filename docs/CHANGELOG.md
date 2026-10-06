# SignLnk — Changelog

Newest first. One entry per session or merged step: what was built, where, measured results
(with method), follow-ups. `/session-close` adds entries. Moved out of PROJECT_CONTEXT.md on
2026-10-05 so the always-loaded context stays small.

- 2026-10-06 — Step 6 merged (PR #6; fresh clip from a second camera passes parity). Step 7 (branch `feat/phase0-adrs`): ADR-0001 landmarks not pixels, ADR-0002 gloss layer, ADR-0003 two data tracks, status proposed; ADR-0003 notes the `track:` CI check is not built yet.
- 2026-10-05 — Claude Code framework added after reading the Claude Code docs (best practices,
  run agents in parallel/subagents, hooks guide, skills, settings): `.claude/settings.json`
  (permissions allow/ask/deny + 4 hooks), hooks `guard-paths`, `post-edit`, `session-context`,
  `stop-gate` (Node, exec form; tested with synthetic inputs), subagents `plan-reviewer`,
  `privacy-reviewer`, `license-auditor`, `test-runner`, skills `/verify`, `/step`,
  `/session-close`, `/schema-change`, `/adr`, `/dataset-license`, `/phase-gate`;
  `docs/WORKFLOW.md`; CLAUDE.md slimmed and imports PROJECT_CONTEXT.md; this changelog split out.
  Second pass: path-scoped `.claude/rules/` (6 files), `prompt-router` UserPromptSubmit hook,
  personal `settings.local.json` (`SIGNLNK_DATA_DIR`), and a "start of every task" + "read and use
  when" table in CLAUDE.md. Config staged in `claude-setup/` until renamed to `.claude/` locally
  (remote tools can't write `.claude`).
- 2026-10-05 — Phase 0 step 5 merged (PR #5). Step 6 built (branch `feat/phase0-record`): `/dev/record` (fps counter, 3 s capture, `.npy` download), TS `.npy` writer + `Recorder`, golden `.npy` shared with the Python loader; a page-downloaded 3 s file round-trips into `load_recording`. First real recording passed all geometry checks but failed parity (webcam 4:3 vs Kaggle portrait aspect); fixed with `reference_aspect=0.78` in the layout, applied in the worker (ADR-0007, `aspect.ts`/`aspect.py` + golden cases), calibrated on 7 recordings from one webcam. Serve-side parity now passes on 6 usable clips of several people (face xy mean 0.054–0.111 vs limit 0.20). MediaPipe's hand label was wrong in ~7% of single-hand frames, so hands are now assigned to the nearer pose wrist (ADR-0008). Still `[VERIFY]`: a different camera; fresh clips with wrist assignment.
- 2026-10-04 — Phase 0 step 4 merged (PR #4). Step 5 built (branch `feat/phase0-islr`): Kaggle ISLR loader (Holistic → slk-landmarks-v1 via `legacy_holistic_indices`, identity), geometry checks, parity stats; 525-sequence sample (21 signers, 311 MB) in `D:\signlnk-data`; Kaggle licence verified (ADR-0006, `docs/datasets.md`). Serve-side parity waits for a `/dev/record` recording (step 6).
- 2026-10-04 — Phase 0 step 3 merged (PR #3). Step 4 built (branch `feat/phase0-normalization`): neck-centred, shoulder-width-scaled normalization in TS and Python, z kept, invalid frames all NaN (ADR-0005); parity ≤1e-5 on 3 golden fixtures, mutation-checked.
- 2026-10-04 — Phase 0 steps 1–2 merged (PR #1, #2): monorepo, CI, five v1 schemas, TS+Pydantic generation with staleness check. Repo now at `C:\dev\signlnk` (outside OneDrive).
- 2026-10-04 — Phase 0 step 3 built (branch `feat/phase0-landmarks`): `slk-landmarks-v1` N=146 (ADR-0004), MediaPipe worker pipeline, `/dev/landmarks` benchmark page. fps gate passed with a real signer: 35.1 fps, worker p95 56.6 ms (pose+face every 2nd frame). Handedness labels used as-is (a swap was wrong).
- 2026-10-04 — Toolkit audited; `docs/TOOLKIT.md` added and linked from CLAUDE.md; ecc + engineering plugins active, qodo optional, miro off, no connectors needed.
- 2026-10-01 — Decisions filled; PLAN.md v1 and CLAUDE.md written; phase corrected to 0.
