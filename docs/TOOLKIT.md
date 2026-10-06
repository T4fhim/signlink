# SignLnk — Toolkit & Standby Roster
Audited 2026-10-05 (previous: 2026-10-04) · Current phase: **0 — Setup** · Re-audit at each phase boundary (`/phase-gate`).

Legend: ✅ **ACTIVE** (on standby now, Phase 0) · 🕒 **LATER** (phase noted) · ⚪ **OPTIONAL** · ⛔ **NOT NEEDED**

Order of preference when two tools do the same job: **project `.claude/` item → built-in Claude Code
command → plugin (ECC / engineering)**. Project items know SignLnk's rules; plugins don't.

---

## 1. Environment status

| Where | Item | Status | Note |
|---|---|---|---|
| Laptop (Windows 11, `tafhim-work`) | Repo `C:\dev\signlnk` (outside OneDrive), git, GitHub remote `origin` | ✅ | PR per Phase step |
| Laptop — Windows host | Node ≥24, pnpm 12.3.4 (`packageManager`), uv, Python 3.11, Git (+ Git Bash) | ✅ (from `package.json`, CI and working builds) | Git Bash runs hook/skill shell commands |
| Laptop — Windows host | Docker Desktop | 🕒 Phase 1 step 8 | API + Postgres |
| Laptop — Cowork Linux VM | git, uv, Node 22, Python 3.11, jq, ffmpeg | ✅ | No pnpm; npm registry blocked → JS installs on Windows only |
| Data | `D:\signlnk-data` (`SIGNLNK_DATA_DIR`) | ✅ | Kaggle ISLR sample; recordings. Never in the repo |
| Cloud workspace | git, Python, uv, Node, pnpm, Docker, gh | ✅ | `gh` token invalid; only needed to push from the cloud |
| Local MCP | `chrome-devtools` (ecc plugin) | ✅ | fps traces, Lighthouse (Phase 1) |

---

## 2. Project Claude Code config (`.claude/`, committed) — ✅ ACTIVE

Loads in every Claude Code session started in the repo, including cloud sessions. Details and
reasoning: `docs/WORKFLOW.md` §8.

| Kind | Item | Job |
|---|---|---|
| Settings | `.claude/settings.json` | Permission allow/ask/deny rules; registers the hooks |
| Hook | `guard-paths.mjs` (PreToolUse) | Blocks edits to generated types, lockfiles, MediaPipe assets, `.env`, recordings/landmark data, NC data in `lexicon/core/` |
| Hook | `post-edit.mjs` (PostToolUse) | Prettier / ruff format; rule reminders (schemas, normalization, layout, `track:`, deps) |
| Hook | `session-context.mjs` (SessionStart) | Branch, dirty files, current phase; hard rules after compaction |
| Hook | `stop-gate.mjs` (Stop) | Blocks on stale generated types or missing `track:` |
| Hook | `prompt-router.mjs` (UserPromptSubmit) | Points a matching prompt to the right skill/subagent/gate; flags later-phase work |
| Rules | `.claude/rules/{landmarks,schemas,ml,web-privacy,lexicon-data,claude-config}.md` | Path-scoped instructions that load only when matching files are read or edited (WORKFLOW §8) |
| Subagent | `plan-reviewer` (Opus) | Fresh-context review of a step vs Done-when, hard rules, scope |
| Subagent | `privacy-reviewer` (Sonnet) | Egress, third-party scripts, storage, consent |
| Subagent | `license-auditor` (Sonnet) | Dependency licences, data tracks, lexicon contamination |
| Subagent | `test-runner` (Haiku) | Runs checks, returns failures only |
| Skill | `/verify` | CI-equivalent suite via `test-runner`; overrides the bundled `/verify`; run before each commit |
| Skill | `/step <p.s>` | Full PLAN step procedure |
| Skill | `/session-close` | Updates PROJECT_CONTEXT, CHANGELOG, CLAUDE.md commands; prints Log to context |
| Skill | `/schema-change` | Add-only schema procedure |
| Skill | `/adr` | Next ADR from template |
| Skill | `/dataset-license` | Licence verification + `docs/datasets.md` entry (Claude may invoke it on its own) |
| Skill | `/phase-gate <n>` | Phase-boundary review on Opus |
| Personal | `.claude/settings.local.json` (gitignored) | Your overrides, e.g. `"env": {"SIGNLNK_DATA_DIR": "D:\\signlnk-data"}` |

🕒 LATER project additions (create when the phase starts, not before — each description costs context):
`ml-eval-auditor` subagent (Phase 1: signer-independent splits, leakage, metric reporting),
`/train-run` skill (Phase 1: Kaggle/Colab run checklist + `track:`), a11y review via the ECC agent (Phase 1 UI).

## 3. Built-in Claude Code commands to use

| Command | When |
|---|---|
| `Shift+Tab` → plan mode, `Ctrl+G` edit plan | Before multi-file or uncertain steps |
| `/clear`, `/compact <focus>`, `Esc Esc` / `/rewind` | Context hygiene; partial summarise; undo Claude's edits |
| `/btw` | Side question that shouldn't enter history |
| `/rename`, `claude --continue`, `claude --resume` | One named session per step |
| `/context` | See what loads every turn (CLAUDE.md, skills, agents, MCP) |
| `/code-review` | Bundled diff review for bugs (complements `plan-reviewer`) |
| `/simplify`, `/debug` | Bundled cleanup and debugging skills |
| `/goal <condition>` | Keep working until a condition holds (unattended runs) |
| `/batch <instruction>` | Mechanical change across many files (worktree subagents) |
| `/tasks` | Check/stop background subagents |
| `/hooks`, `/status`, `/config`, `/permissions`, `/sandbox` | Inspect hooks, settings sources, permission rules |
| `/doctor`, `claude doctor`, `/skill-doctor` | CLAUDE.md cuts, rejected settings, unused-skill cost (phase gates) |

## 4. Plugins

| Plugin | Status | Why |
|---|---|---|
| **ecc** (68 agents, ~290 skills, hooks, chrome-devtools MCP) | ✅ ACTIVE | Language reviewers, build fixers, TDD, perf. **Context cost**: its skill and agent descriptions load every turn; run `/skill-doctor` at each phase gate and disable unused ones via `/plugin` |
| **engineering** | ✅ ACTIVE | ADR/system-design/testing-strategy helpers |
| **qodo** | ⚪ OPTIONAL — off by default | Needs qodo CLI + account; diffs go to Qodo's cloud. Free tier/terms **[VERIFY]** |
| **miro** | ⛔ NOT NEEDED | Mermaid in-repo covers diagrams |

Overlaps, and which to use:
- Session save/resume: built-in `/rename` + `--resume` + `/session-close` over `/ecc:save-session` / `/ecc:resume-session`.
- Checkpoints: built-in `/rewind` + git over `/ecc:checkpoint`.
- Review: `plan-reviewer` (plan conformance) + `/code-review` (bugs); ECC language reviewers for depth.
- Formatting: project `post-edit.mjs` hook. If an ECC hook also formats on edit, disable one (check `/hooks`) **[VERIFY]**.
- ECC **GateGuard** asks for "facts" before the first Bash/Write per file. Keep it if useful; it slows multi-file work. Turn off with `ECC_GATEGUARD=off` or scope it with `GATEGUARD_EXEMPT_GLOBS` (e.g. `.claude/**,docs/**`).

User-level plugins don't load in cloud sessions; only plugins enabled in the repo's
`.claude/settings.json` do. SignLnk doesn't enable plugins at project level (contributors shouldn't
need ECC).

## 5. MCP servers & connectors

| Item | Status | Use |
|---|---|---|
| chrome-devtools (local, ecc) | ✅ ACTIVE | `performance_start_trace` for ≥25 fps; `lighthouse_audit` for PWA (Phase 1) |
| Built-in browser / Claude in Chrome | ✅ ACTIVE | Dataset licence **[VERIFY]** pages, Kaggle rules, MediaPipe docs |
| Claude Docs | ✅ ACTIVE | Living docs: consent form, review checklists, advisor outreach |
| Projects (SignLink) | ✅ ACTIVE | Mirrors of the repo docs (repo is canonical, WORKFLOW §12) |
| Scheduled tasks | ⚪ OPTIONAL | e.g. weekly "re-check dataset licences & lib versions" |
| **Context7** (library docs MCP) | ⚪ RECOMMENDED ADD | Free; current MediaPipe Tasks / onnxruntime-web / Next.js docs; `ecc:docs-lookup` needs it. `claude mcp add context7 -- npx -y @upstash/context7-mcp` **[VERIFY package name at install]** |
| Playwright MCP | ⛔ NOT NEEDED | `@playwright/test` in CI is enough |
| Connectors (Brevo, Canva, Fathom, Figma, Gamma, Indeed, Krisp, Microsoft 365, Spotify, Todoist) | ⛔ NOT NEEDED | Figma ⚪ maybe for Studio UI (Phase 1). **Never** use Fathom/Krisp on advisor or signer sessions (recording/privacy) |
| GitHub | `gh` CLI | Context-efficient; no connector needed |

## 6. Subagents (plugin + built-in)

### ✅ ACTIVE — Phase 0
| Agent | Phase 0 job |
|---|---|
| Project agents (§2) | First choice for review, privacy, licences, test runs |
| `Explore`, `Plan`, `general-purpose` (built-in) | Read-only search / planning research / multi-step side tasks |
| `ecc:architect`, `ecc:code-architect` | Layout/interface design, ADRs |
| `ecc:type-design-analyzer` | JSON Schemas (add-only rule) |
| `ecc:tdd-guide` | Golden-fixture parity tests |
| `ecc:python-reviewer`, `ecc:typescript-reviewer`, `ecc:react-reviewer` | Language-level review |
| `ecc:build-error-resolver`, `ecc:react-build-resolver` | pnpm/Next.js/TS build breaks |
| `ecc:performance-optimizer` | ≥25 fps |
| `ecc:silent-failure-hunter` | NaN handling for missing landmarks |
| `ecc:mle-reviewer` | Train/serve parity design |
| `ecc:doc-updater` | Codemaps (CLAUDE.md commands are updated by `/session-close`) |

### 🕒 LATER
| Agent | Phase |
|---|---|
| `ecc:pytorch-build-resolver`, `ecc:mle-reviewer` (training) | 1 |
| `ecc:fastapi-reviewer`, `ecc:database-reviewer` | 1 (API + Postgres, Studio) |
| `ecc:a11y-architect` | 1 (demo + Studio UI — high priority for this audience) |
| `ecc:e2e-runner` | 1 (Playwright smoke test in CI) |
| `ecc:opensource-sanitizer`, `ecc:opensource-packager` | 1 step 13 (public demo) |
| `ecc:docs-lookup` | When Context7 is added |
| `ecc:rust-reviewer`, `ecc:rust-build-resolver` | 6 (Tauri) |

### ⛔ NOT NEEDED
Go, Java, Kotlin, C++, C#, F#, PHP, Swift, Dart/Flutter, Django, Vue, HarmonyOS reviewers/resolvers; healthcare, network/homelab, marketing, SEO, chief-of-staff, GAN harness agents, RAG reviewer. Agent teams (experimental) — not used.

## 7. Plugin skills & slash commands

### ✅ ACTIVE — Phase 0
| Group | Skills / commands |
|---|---|
| Workflow | Project `/step`, `/verify`, `/session-close` first; `/ecc:plan`, `/ecc:prp-plan`, `/ecc:feature-dev`, `/ecc:model-route` when useful |
| Design | Project `/adr` + `engineering:architecture` / `ecc:architecture-decision-records`; `ecc:contract-first` (schemas); `engineering:system-design` |
| Python | `ecc:python-patterns`, `ecc:python-testing`, `/ecc:python-review` |
| Frontend | `ecc:nextjs-turbopack`, `ecc:react-patterns`, `ecc:frontend-patterns`, `ecc:react-testing`, `/ecc:react-review`, `/ecc:react-build` |
| Quality | `ecc:tdd-workflow`, `engineering:testing-strategy`, `/ecc:build-fix`, `/ecc:test-coverage` |
| Perf | `ecc:benchmark-methodology`, `ecc:latency-critical-systems` |
| Repo | `ecc:git-workflow`, `ecc:github-ops`, `/ecc:update-codemaps` |
| Research | Project `/dataset-license`; `ecc:search-first`, `ecc:deep-research`, `ecc:documentation-lookup` |
| Docs | `anthropic-skills:docs` (living docs), `dataviz` (fps/latency charts) |

### 🕒 LATER
| Skill | Phase |
|---|---|
| `ecc:pytorch-patterns`, `ecc:mle-workflow`, `ecc:eval-harness` | 1 (training + signer-independent eval) |
| `ecc:fastapi-patterns`, `/ecc:fastapi-review`, `ecc:api-design`, `ecc:postgres-patterns`, `ecc:database-migrations`, `ecc:docker-patterns`, `ecc:backend-patterns` | 1 (Studio API) |
| `ecc:accessibility`, `ecc:frontend-a11y`, `ecc:security-review`, `ecc:e2e-testing`, `ecc:browser-qa` | 1 |
| `ecc:opensource-pipeline`, `engineering:deploy-checklist` | 1 step 13 |
| `ecc:blender-motion-state-inspection`, `ecc:motion-*` | 5 (avatar) |
| `ecc:rust-patterns`, `ecc:windows-desktop-e2e` | 6 (Tauri) |

### ⛔ NOT NEEDED
All language packs outside Python/TS (Rust until P6); healthcare/HIPAA; network/homelab; finance, trading, DeFi, logistics, marketing, social, investor, SEO, email/messages ops; `morning`, `import-memory`, `google-workspace`; `pptx`/`docx`/`xlsx`/`pdf` only on request. These still cost context while installed — see §4.

---

## 8. User-level setup (your machine, not committed)

| Item | Where | Note |
|---|---|---|
| Data dir | `.claude/settings.local.json` → `"env": {"SIGNLNK_DATA_DIR": "D:\\signlnk-data"}` | Lets tests and hooks find real data |
| Desktop notification when Claude waits | `%USERPROFILE%\.claude\settings.json`, `Notification` hook | Docs' Windows example uses `powershell.exe` + `System.Windows.Forms.MessageBox` (a dialog, not a toast). Personal preference, so not in the repo |
| Fewer prompts | `/permissions` or `settings.local.json` allow rules | Keep `deny` rules in the shared file |
| Optional privacy setting | `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` in user settings `env` | Disables Claude Code's own non-essential traffic (telemetry, error reports, auto-update); your choice |

## 9. Accounts needed (all free)
| Account | When | Note |
|---|---|---|
| GitHub | ✅ in use | Repo, Actions CI, Pages |
| Kaggle (rules accepted, token in `%USERPROFILE%\.kaggle\`) | ✅ in use | Dev-time only, never a runtime dependency. Claude is denied reading `~/.kaggle/` |
| Colab | Phase 1 | Training fallback |

## 10. Resolved since the 2026-10-04 audit
- Repo moved to `C:\dev\signlnk` outside OneDrive; GitHub remote in use, PRs #1–#5 merged.
- Phase 0 runs in Claude Code on Windows (as recommended).
- Benchmark laptop named (PROJECT_CONTEXT: i5-1135G7, 15.7 GB, Iris Xe, Windows 11).
- Project `.claude/` config added (§2).
