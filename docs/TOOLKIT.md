# SignLnk — Toolkit & Standby Roster
Audited 2026-10-04 · Current phase: **0 — Setup** · Re-audit at each phase boundary.

Legend: ✅ **ACTIVE** (on standby now, Phase 0) · 🕒 **LATER** (phase noted) · ⚪ **OPTIONAL** · ⛔ **NOT NEEDED**

---

## 1. Environment status

| Where | Item | Status | Action |
|---|---|---|---|
| Laptop (Windows, `tafhim-work`) | Project folder `OneDrive\Documents\Projects\SignLink` | Docs only (CLAUDE.md, PROJECT_CONTEXT.md, docs/PLAN.md, .gitignore, .gitattributes). **No git repo yet.** | See §6 decision 1 |
| Laptop — Cowork Linux VM | git 2.34, uv 0.12, Node 22, ffmpeg 4.4, **Python 3.11.16 installed via uv today** | ✅ | — |
| Laptop — Cowork Linux VM | pnpm, Docker, gh | ❌ npm registry blocked by egress; no Docker | Use Windows host or cloud workspace for JS installs |
| Laptop — Windows host | Git / Python / Node / pnpm / Docker | **Unknown** (not visible from here) | Run §6 check |
| Cloud workspace | git 2.43, Python 3.13, uv 0.11, Node 22, pnpm 10.28, Docker 29.8, gh 2.89, ffmpeg 6.1 | ✅ full toolchain | — |
| Cloud workspace | GitHub CLI auth | ❌ token invalid | Needed only if Claude pushes from the cloud |
| Local MCP | `chrome-devtools` (ecc plugin) | ✅ announced, responding | Used for fps traces + Lighthouse |
| Hooks | ECC **GateGuard** (fact-forcing before first Bash/Write) | Active | Keep (cheap safety); disable with `ECC_GATEGUARD=off` if it slows you |

---

## 2. Plugins

| Plugin | Status | Why |
|---|---|---|
| **ecc** (68 agents, ~290 skills, hooks, chrome-devtools MCP) | ✅ ACTIVE | Core dev harness: planning, TDD, reviewers, build fixers |
| **engineering** | ✅ ACTIVE | ADRs, testing strategy, system design, debug, docs |
| **qodo** | ⚪ OPTIONAL — off by default | Needs qodo CLI + account; diffs go to Qodo's cloud. Free tier/terms **[VERIFY]**. ecc reviewers cover this at $0 |
| **miro** | ⛔ NOT NEEDED | Needs Miro account; Mermaid in-repo covers diagrams |

## 3. MCP servers & connectors

| Item | Status | Use |
|---|---|---|
| chrome-devtools (local, ecc) | ✅ ACTIVE | `performance_start_trace` for ≥25 fps check; `lighthouse_audit` for PWA (Phase 1) |
| Built-in browser / Claude in Chrome | ✅ ACTIVE | Dataset license **[VERIFY]** pages, Kaggle rules, MediaPipe docs |
| Claude Docs | ✅ ACTIVE | Living docs: consent form, review checklists, advisor outreach |
| Projects (SignLink) | ✅ ACTIVE | PROJECT_CONTEXT / PLAN / CLAUDE.md sync |
| Scheduled tasks | ⚪ OPTIONAL | e.g. weekly "re-check dataset licenses & lib versions" |
| **Context7** (library docs MCP) | ⚪ RECOMMENDED ADD | Free; `ecc:docs-lookup` agent needs it. Gives current MediaPipe Tasks / onnxruntime-web / Next.js docs. Add in Claude Code: `claude mcp add context7 -- npx -y @upstash/context7-mcp` **[VERIFY package name at install]** |
| Playwright MCP | ⛔ NOT NEEDED | `@playwright/test` in CI is enough |
| Connectors: Brevo, Canva, Fathom, Figma, Gamma, Indeed, Krisp, Microsoft 365, Spotify, Todoist | ⛔ NOT NEEDED (all disconnected) | Figma ⚪ maybe for Studio UI (Phase 1). **Do not** use Fathom/Krisp on advisor or signer sessions (recording/privacy) |
| GitHub connector | Not in registry | Use `gh` CLI instead |

---

## 4. Subagents (Agent tool)

### ✅ ACTIVE — Phase 0
| Agent | Phase 0 job |
|---|---|
| `Plan`, `ecc:planner` | Break each Phase 0 step into a checklist before coding |
| `ecc:architect`, `ecc:code-architect` | Landmark layout `slk-landmarks-v1`, ADR-0001/2/3 |
| `ecc:type-design-analyzer` | JSON Schemas in `packages/schemas` (add-only rule) |
| `ecc:tdd-guide` | Golden-fixture parity tests (TS ↔ Py ≤1e-5) |
| `ecc:python-reviewer` | `ml/signlnk_ml/features`, Kaggle loader |
| `ecc:typescript-reviewer`, `ecc:react-reviewer` | `packages/landmarks`, Web Worker, `/dev/record` |
| `ecc:build-error-resolver`, `ecc:react-build-resolver` | pnpm/Next.js/TS build breaks |
| `ecc:performance-optimizer` | ≥25 fps on benchmark laptop |
| `ecc:silent-failure-hunter` | NaN-handling for missing landmarks |
| `ecc:security-reviewer` | No third-party scripts, camera data stays on device |
| `ecc:mle-reviewer` | Train/serve parity test design |
| `ecc:code-reviewer` | Every change |
| `ecc:doc-updater` | Keep CLAUDE.md commands + codemaps current |
| `Explore`, `general-purpose` | Search / research fan-out |

### 🕒 LATER
| Agent | Phase |
|---|---|
| `ecc:pytorch-build-resolver` | 1 (training) |
| `ecc:fastapi-reviewer`, `ecc:database-reviewer` | 1 (API + Postgres, Studio) |
| `ecc:a11y-architect` | 1 (demo + Studio UI — high priority for this audience) |
| `ecc:e2e-runner` | 1 (Playwright smoke test in CI) |
| `ecc:opensource-sanitizer`, `ecc:opensource-packager` | 1 step 13 (public demo release) |
| `ecc:docs-lookup` | As soon as Context7 is added |
| `ecc:rust-reviewer`, `ecc:rust-build-resolver` | 6 (Tauri) |

### ⛔ NOT NEEDED
Go, Java, Kotlin, C++, C#, F#, PHP, Swift, Dart/Flutter, Django, Vue, HarmonyOS reviewers/resolvers; healthcare, network/homelab, marketing, SEO, chief-of-staff, GAN harness agents, RAG reviewer.

---

## 5. Skills & slash commands

### ✅ ACTIVE — Phase 0
| Group | Skills / commands |
|---|---|
| Workflow | `/ecc:plan`, `/ecc:prp-plan`, `/ecc:orch-build-mvp` (scaffold from PLAN.md), `/ecc:orch-add-feature`, `/ecc:feature-dev`, `/ecc:checkpoint`, `/ecc:save-session`, `/ecc:resume-session`, `/ecc:model-route` |
| Design | `engineering:architecture` + `ecc:architecture-decision-records` (ADRs), `ecc:contract-first` (schemas), `engineering:system-design` |
| Python | `ecc:python-patterns`, `ecc:python-testing`, `/ecc:python-review` |
| Frontend | `ecc:nextjs-turbopack`, `ecc:react-patterns`, `ecc:frontend-patterns`, `ecc:react-testing`, `/ecc:react-review`, `/ecc:react-build` |
| Quality | `ecc:tdd-workflow`, `ecc:verification-loop`, `engineering:testing-strategy`, `/ecc:build-fix`, `/ecc:test-coverage`, `/ecc:code-review` |
| Perf | `ecc:benchmark-methodology`, `ecc:latency-critical-systems` (fps + latency budget) |
| Repo | `ecc:git-workflow`, `ecc:github-ops`, `/ecc:update-docs`, `/ecc:update-codemaps` |
| Research | `ecc:search-first`, `ecc:deep-research` (all **[VERIFY]** items), `ecc:documentation-lookup` |
| Docs | `anthropic-skills:docs` (living docs), `dataviz` (fps/latency charts) |

### 🕒 LATER
| Skill | Phase |
|---|---|
| `ecc:pytorch-patterns`, `ecc:mle-workflow`, `ecc:eval-harness` | 1 (model training + signer-independent eval) |
| `ecc:fastapi-patterns`, `/ecc:fastapi-review`, `ecc:api-design`, `ecc:postgres-patterns`, `ecc:database-migrations`, `ecc:docker-patterns`, `ecc:backend-patterns` | 1 (Studio API) |
| `ecc:accessibility`, `ecc:frontend-a11y`, `ecc:security-review`, `ecc:e2e-testing`, `ecc:browser-qa` | 1 |
| `ecc:opensource-pipeline`, `engineering:deploy-checklist` | 1 step 13 |
| `ecc:blender-motion-state-inspection`, `ecc:motion-*` | 5 (avatar) |
| `ecc:rust-patterns`, `ecc:windows-desktop-e2e` | 6 (Tauri) |

### ⛔ NOT NEEDED
All language packs outside Python/TS (Go, Rust until P6, Kotlin, Java/Spring/Quarkus, C++, Laravel, Django, Rails, Perl, Vue/Nuxt/Angular, Flutter, Swift); healthcare/HIPAA; network/homelab; finance, trading, DeFi, logistics, marketing, social, investor, SEO, email/messages ops; `morning`, `import-memory`, `google-workspace`, `pptx`/`docx`/`xlsx`/`pdf` (only on request).

---

## 6. Open decisions before code (blocking Phase 0 step 1)

1. **Repo location.** The folder is inside **OneDrive**. Git + `node_modules` + `.venv` + MediaPipe WASM inside OneDrive causes sync churn and file-lock errors. Recommended: repo at `C:\dev\signlnk` (outside OneDrive), pushed to GitHub; keep OneDrive only for exports.
2. **GitHub repo.** Name + public/private now (Apache-2.0; plan is public at the Phase 1 demo).
3. **Where Phase 0 runs.** Recommended: **Claude Code on Windows** (CLAUDE.md marks Phase 0 as 🤖 Claude Code). Cowork's device VM can't reach the npm registry, so it can't do `pnpm install`.
4. **Benchmark laptop** still TBD (needed for the ≥25 fps gate). If it's this machine, record CPU/RAM/GPU.

### Windows host check (PowerShell, run once)
```powershell
git --version; node --version; python --version; uv --version; pnpm --version; docker --version; gh --version
# Install anything missing (all free):
winget install --id Git.Git -e
winget install --id OpenJS.NodeJS.LTS -e
winget install --id astral-sh.uv -e
corepack enable; corepack prepare pnpm@latest --activate
uv python install 3.11
winget install --id GitHub.cli -e        # then: gh auth login
# Docker Desktop only needed from Phase 1 step 8 (API + Postgres)
```
Pin exact versions in lockfiles at scaffold time; check current stable versions then.

---

## 7. Accounts needed (all free)
| Account | When | Note |
|---|---|---|
| GitHub | Phase 0 | Repo, Actions CI, Pages |
| Kaggle (+ accept ISLR competition rules, API token in `%USERPROFILE%\.kaggle\kaggle.json`) | Phase 0 step 5 | Dev-time only. Never a runtime dependency. Competition terms **[VERIFY]** |
| Colab | Phase 1 | Training fallback |
