# SignLnk — How we work with Claude

Version 1 · 2026-10-05 · Source: Claude Code docs (best practices, run agents in parallel,
subagents, hooks guide, skills, settings), read 2026-10-05. Features marked *(vX)* need that
Claude Code version or later; check with `claude --version`. Where the docs and this file
disagree, the docs win; fix this file.

This file holds the procedures. `CLAUDE.md` holds only what every session needs; it links here.

---

## 1. Where each kind of work happens

| Work | Where | Why |
|---|---|---|
| Architecture, phase reviews, translation design, "should we…" | claude.ai project chat (Opus) or Claude Code on Opus | Judgement; no repo writes needed |
| Multi-file implementation, tests, git | **Claude Code on Windows** in `C:\dev\signlnk` | Full toolchain; project `.claude/` config loads |
| Docs, research, licence checks, file reorganisation | Cowork or Claude Code | Either works |
| Long unattended jobs (training runs, fan-out) | Kaggle/Colab for training; `claude -p` locally for fan-out | $0 compute |

Cloud sessions (claude.ai/code) read only the committed `.claude/settings.json`, project skills and
CLAUDE.md from the clone. User settings, `settings.local.json`, personal skills and plugins enabled
only in user settings (ECC, engineering) **don't** load there. Cowork's Linux VM can't reach the
npm registry, so `pnpm install` must run on Windows.

## 2. Session lifecycle

1. **Start** in the repo root (`C:\dev\signlnk`) so project settings, hooks, agents and skills
   load. The SessionStart hook injects branch, dirty files and the current phase line.
2. **Name** the session for the step: `/rename p0-s6-record`. Resume later with
   `claude --continue` or `claude --resume`. One workstream per session, like a branch.
3. **Work** one PLAN step per session with `/step <phase.step>` (§3).
4. **Clear** with `/clear` between unrelated tasks. Never mix a step with an unrelated question;
   use `/btw` for a side question (its answer never enters history).
5. **Close** with `/session-close`: it updates PROJECT_CONTEXT, CHANGELOG and CLAUDE.md commands and
   prints `Log to context:` + `Next step:`.

## 3. The step procedure (`/step`)

Explore → plan → implement → verify → review → commit. The `/step` skill runs it; the reasoning:

- **Scope first.** The step's **Done when** line is the acceptance test. Later-phase work is out.
- **Plan only when it pays.** Use plan mode (`Shift+Tab` until `⏸ plan mode on`, or
  `claude --permission-mode plan`) when the approach is uncertain, several files change, or the
  code is unfamiliar. Edit the plan with `Ctrl+G`. If the diff fits in one sentence, skip the plan.
- **Specific prompts.** Name files with `@path`, the symptom, the constraint, and the pattern to
  follow (for example "mirror `packages/landmarks/src/normalize.ts` in
  `ml/signlnk_ml/features/normalize.py`; add a golden fixture case; run both parity suites").
- **Tests first** for anything with a measurable Done-when (parity, fps, latency, accuracy).
- **Evidence over assertion.** Every summary shows the commands run and their output, or the
  measured number and how it was measured.
- **Commit small**, Conventional Commits, on `feat/phase<P>-<slug>`. PR to `main`; CI must pass.

For a large feature (Studio v1, decoder, translation baseline): first run the interview pattern —
"Interview me in detail using the AskUserQuestion tool about <feature>… then write a complete spec
to `docs/specs/<feature>.md`" — then start a **fresh session** to implement the spec. A good spec
names files and interfaces, states what's out of scope, and ends with an end-to-end verification
step.

## 4. Verification ladder

Each rung trades setup for attention. Use the lowest rung that makes the result trustworthy.

| Rung | Mechanism in this repo | Use for |
|---|---|---|
| 1. In the prompt | "…run the tests and iterate until they pass" | Any task |
| 2. Skill | `/verify` (test-runner subagent; same gates as CI). A project skill named `verify` is run before each commit *(v2.1.286)* | Every commit |
| 3. Session goal | `/goal <condition>` re-checked after every turn | Unattended multi-turn work |
| 4. Deterministic gate | Stop hook `.claude/hooks/stop-gate.mjs` (stale generated types, missing `track:`) | Rules that must never slip |
| 5. Second opinion | `plan-reviewer` subagent (Opus, fresh context) or bundled `/code-review` | End of every step |
| 6. Humans | CI on the PR; **community gates G0–G5** (PLAN §9) | Anything Deaf users will see |

Rung 6 is never automated: nothing shown to Deaf users is "correct" until fluent/Deaf signers
sign off in `docs/review/`.

Reviewers asked to find gaps always find some. Tell them to report only gaps that affect
correctness, a hard rule or the step's requirements; treat the rest as optional, or you'll
over-engineer.

## 5. Context management

The context window is the scarcest resource; performance drops as it fills.

- `/clear` between unrelated tasks. After **two failed corrections** on the same issue, `/clear`
  and restart with a better prompt that includes what you learned.
- Delegate wide reading: "use a subagent to investigate how the worker handles dropped frames".
  `test-runner` keeps test output out of the main context.
- `/compact Focus on <topic>` for manual compaction; `Esc Esc` / `/rewind` → *Summarize from here*
  for partial compaction. The SessionStart hook re-injects the hard rules after any compaction.
- `/context` shows what's loaded (CLAUDE.md, skills, agents, MCP). ECC adds ~290 skills and 68
  agents; their descriptions cost context every turn. Run `/skill-doctor` *(v2.1.252)* at each
  phase gate and turn off plugin skills you never use via `/plugin`.
- **Checkpoints** (`Esc Esc`, `/rewind`) undo Claude's file edits only. Changes made by commands
  (`pnpm gen:types`, `write-layout`, fixture generators) and by background forked skills are not
  captured. Commit before risky steps; git is the real undo.

## 6. Prompt patterns that work here

| Instead of | Say |
|---|---|
| "fix the parity test" | "test_geometry_parity fails on face landmarks after the aspect change; check `aspect.ts` vs `aspect.py`, write a failing golden case, then fix" |
| "make it faster" | "worker p95 is 56.6 ms on the benchmark laptop; profile with chrome-devtools `performance_start_trace` on `/dev/landmarks` and propose changes that keep ≥25 fps" |
| "add the Kaggle data" | "/dataset-license Kaggle ISLR, then extend `kaggle_islr.py` to load N sequences per signer" |
| "why is this weird?" | "look at the git history of `layoutSpec.ts` and summarise how the index subset came to be" |

## 7. Parallel work

| Need | Tool |
|---|---|
| Side research, long test output | Subagents (`Explore`, `test-runner`, reviewers). Keep to 2–3 at once when all results return to the main session |
| Two independent tracks (e.g. Studio API and model training in Phase 1) | Separate sessions in **git worktrees** (`claude --worktree` or `isolation: worktree`), each on its own branch |
| Fresh-eyes review | Writer/Reviewer: session A implements, session B (or `plan-reviewer`) reviews the diff |
| Same mechanical change across many files | `/batch <instruction>` (5–30 worktree subagents) or a `claude -p` loop with `--allowedTools` and `--permission-mode dontAsk`; test on 2–3 files first |
| Hand off and check back | Agent view (`claude agents`, research preview) |

Not used: agent teams (experimental, high token cost) and dynamic workflows unless a task clearly
outgrows subagents. Parallel sessions multiply token use.

## 8. Permissions and safety

- **Modes.** Auto mode is the default interactive mode *(v2.1.283)*: a classifier blocks risky
  actions. Use plan mode for exploration, `dontAsk` for scripted `claude -p` runs, Manual when you
  want to approve every edit. `auto` and `bypassPermissions` can't be set from project settings.
- **Project rules** live in `.claude/settings.json` (committed): routine pnpm/uv/git read and
  commit commands are allowed; push, reset, rebase, dependency adds/removes, docker and edits to
  `lexicon/`, `docs/datasets.md` and `LICENSE` ask; reading `.env` and `~/.kaggle/` and force
  pushes are denied. Allow rules apply after you trust the folder; deny and ask rules apply at once.
- **Personal overrides** go in `.claude/settings.local.json` (gitignored), for example
  `"env": {"SIGNLNK_DATA_DIR": "D:\\signlnk-data"}` or extra allow rules.
- **Hooks enforce rules deterministically** (advice in CLAUDE.md can be ignored; hooks can't):

| Hook | Event | Enforces |
|---|---|---|
| `guard-paths.mjs` | PreToolUse Edit/Write | No hand edits to generated types, lockfiles, MediaPipe assets, `.env`; no recordings/landmark data in the repo; no ASL-LEX/Signbank data in `lexicon/core/` |
| `post-edit.mjs` | PostToolUse Edit/Write | Prettier/ruff format; reminders for schema (rule 5), normalization (rule 6), layout, `track:` (rule 4), dependency licences (rule 1) |
| `session-context.mjs` | SessionStart | Branch, dirty files, phase; hard rules after compaction |
| `prompt-router.mjs` | UserPromptSubmit | Silent unless the prompt matches a situation; then points to the skill/subagent/gate (datasets, schemas, commits, steps, phase gates, session end, decisions, Deaf-facing output, privacy, dependencies) and flags later-phase work |
| `stop-gate.mjs` | Stop | Blocks on stale generated types or a config without `track:`; reminds on one-sided normalization changes |

- **What loads when.** Every session: `CLAUDE.md` + imported `PROJECT_CONTEXT.md`, skill and
  subagent descriptions, SessionStart output. Every prompt: router pointer when it matches. When
  Claude reads/edits matching files: `.claude/rules/*.md` (path-scoped) and nested
  `apps/web/CLAUDE.md`. On demand: skills (`/name`), subagents, PLAN sections, ADRs.

| Rule file | Loads for |
|---|---|
| `landmarks.md` | `packages/landmarks/**`, `ml/signlnk_ml/{features,data}/**`, layouts, `tests/fixtures/**`, `apps/web/app/dev/**` |
| `schemas.md` | `packages/schemas/**`, generated Python types, `scripts/gen-types.mjs` |
| `ml.md` | `ml/**/*.py`, `ml/configs/**` |
| `web-privacy.md` | `apps/web/**`, worker/pipeline/recorder/protocol, MediaPipe asset scripts |
| `lexicon-data.md` | `lexicon/**`, `docs/datasets.md`, `docs/consent/**`, `docs/review/**` |
| `api.md` | `services/**` |
| `claude-config.md` | `.claude/**`, `CLAUDE.md`, `PROJECT_CONTEXT.md`, `docs/WORKFLOW.md`, `docs/TOOLKIT.md` |

All hooks run as `node <script>` (exec form) so they work on Windows without jq or a shell.
Check them with `/hooks`; debug with `claude --debug-file <path>` or `/debug`; test one by piping
JSON: `echo '{"tool_input":{"file_path":"uv.lock"}}' | node .claude/hooks/guard-paths.mjs`.

## 9. Non-interactive use

`claude -p "<prompt>" --output-format json` is fine for local scripts and fan-out. It is **not** a
CI dependency: CI must stay $0 and key-free, and the product never depends on an LLM at runtime.

## 10. Avoid these

| Pattern | Fix |
|---|---|
| Kitchen-sink session | `/clear` between unrelated tasks |
| Correcting over and over | After two failures, `/clear` and re-prompt with what you learned |
| Bloated CLAUDE.md | Prune; move procedures to skills; turn repeated violations into hooks |
| Trust-then-verify gap | No "done" without `/verify` output and a reviewer pass |
| Infinite exploration | Scope the question or give it to a subagent |
| Chasing every review nit | Fix correctness and requirement gaps only |

## 11. Maintenance cadence

| When | Do |
|---|---|
| Every session | `/session-close` |
| Every step | `/step` → `/verify` → `plan-reviewer` → PR |
| Every phase boundary | `/phase-gate <n>` on Opus: Done-when evidence, community gate, plan review, CLAUDE.md pruning (`/doctor`, `/context`), toolkit re-audit (`/skill-doctor`, `/hooks`, `/status`), licence re-check |
| When Claude ignores a rule twice | Make it a hook, or emphasise that one line with **IMPORTANT** |
| When a setting seems ignored | `/status` (which files loaded), `claude doctor` (rejected entries) |

## 12. Source of truth and the claude.ai project

The **repo is canonical** for `CLAUDE.md`, `PROJECT_CONTEXT.md`, `docs/PLAN.md`, `docs/TOOLKIT.md`,
`docs/WORKFLOW.md` and `docs/CHANGELOG.md`. The claude.ai "SignLink" project holds mirrors so chat
answers stay current. After `/session-close`, re-upload any of those files that changed (or ask
Claude in Cowork to sync them with the Projects tool). If a mirror and the repo disagree, the repo
wins.
