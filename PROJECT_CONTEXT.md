# SignLnk (working name) — Project Context
Last updated: 2026-10-05
Long-form plan: `docs/PLAN.md` · How we work with Claude: `docs/WORKFLOW.md` ·
Toolkit: `docs/TOOLKIT.md` · History: `docs/CHANGELOG.md` · Datasets: `docs/datasets.md`

Keep this file lean: it is imported into every Claude Code session through CLAUDE.md.
Current state and decisions go here; history goes in the changelog.

## Current phase
**Phase 0 — Setup, step 6 of 7** (branch `feat/phase0-record`, PR #6): `/dev/record` built; serve-side parity passes with `reference_aspect=0.78` (ADR-0007) and wrist-based hand assignment (ADR-0008); verified 2026-10-06 on a fresh clip from a second camera. Then step 7 (ADR-0001/2/3 not yet written) and `/phase-gate 0`. Next: Phase 1 — isolated signs + Studio v1.

| Phase 0 step | Status |
|---|---|
| 1 Monorepo + CI | ✅ PR #1 |
| 2 Schemas + type generation | ✅ PR #2 |
| 3 Landmarks worker, ≥25 fps | ✅ PR #3 — 35.1 fps, worker p95 56.6 ms on benchmark laptop |
| 4 TS↔Py normalization parity | ✅ PR #4 — ≤1e-5 on 3 golden fixtures |
| 5 Kaggle ISLR loader + parity stats | ✅ PR #5 |
| 6 `/dev/record` round-trip + serve-side parity | ✅ PR #6 — parity passes on 7 clips incl. a fresh one from a second camera |
| 7 ADR-0001/0002/0003 | ⬜ not started |

## Mission
Real-time, bidirectional, free/open-source communication bridge between signing
and non-signing users. Self-hostable, privacy-first, works without paid APIs.

## Decisions (filled 2026-10-01)
- Target sign language: **ASL (`ase`) only**. Pipeline stays language-agnostic (config value).
- Spoken/written language pair: **ASL ↔ English (`en`)**
- Deaf/signing advisors: **recruit ≥2 fluent Deaf ASL signers before the public demo** (none yet)
- Hardware: **no local GPU assumed → Kaggle / Colab free tiers**; benchmark laptop: **this machine** — i5-1135G7 (4C/8T), 15.7 GB, Iris Xe, Windows 11
- Project license: **Apache-2.0 (code)**; models/data have their own license files
- Commercial use: **not now, door open → two data tracks** (release = permissive only; research = NC allowed)
- Phase 1 vocab: **20–50 meeting-relevant signs**, chosen by signer advisors
- Install model: **one-sided** (works if only the Deaf user has the app)
- First platform: **Chrome/Edge on Windows** → macOS → Linux
- Capture consent: on-device only, visible indicator, chat notice, no recording by default

## Roadmap
| Phase | Goal | Done when |
|---|---|---|
| 0 | Monorepo, schemas, in-browser landmarks, parity tests | ≥25 fps; TS↔Py normalization parity; train/serve parity |
| 1 | Isolated signs (20–50) → live captions; PWA; Signer Studio v1 | ≥85% top-1 unseen signers; p95 <500 ms; gate G1 passed |
| 2 | Continuous signing (CTC) | Gloss WER measured and improving; <500 ms |
| 3 | Speech → gloss → clips; Studio v2 (clips) | Local Whisper; gate G2 passed |
| 4 | Gloss ↔ English translation; Studio v3 (sentence pairs) | chrF/BLEU baseline; gate G3 passed |
| 5 | Pose / 3D avatar output; Studio v4 | Gate G4 passed |
| 6 | Overlay, Tauri desktop, virtual cam, Zoom/Teams caption sinks, WebRTC sessions | Gate G5 meeting pilot |

## Delivery & integration
Website demo → installable PWA → optional Tauri desktop companion.
Meetings: overlay window, virtual camera, local system-audio capture, Zoom caption token / Teams CART link.
Not used: meeting bots, DOM-scraping extensions, Meet Media API (restricted developer preview).

## Stack (all free/OSS)
| Layer | Choice |
|---|---|
| Monorepo | pnpm workspaces + uv; GitHub Actions CI |
| Frontend | Next.js + React + TypeScript, PWA; `/studio` for signers |
| Landmarks | MediaPipe Tasks (hand, pose, face) in a Web Worker (WASM) |
| Model | PyTorch: 1D conv + Transformer encoder → classifier + 256-d embedding (prototypes); ONNX export |
| Inference | onnxruntime-web (browser) / ONNX Runtime |
| Speech | Silero VAD + whisper.cpp / faster-whisper; Piper TTS |
| Backend | FastAPI + WebSockets; PostgreSQL (SQLite in dev); Docker Compose |
| Desktop | Tauri 2 (Phase 6) |
| Sign output | Clips → pose-format sequences → Three.js glTF avatar |
| Tracking | MLflow or CSV |
| Dev assistant | Claude Code with project `.claude/` (settings, hooks, subagents, skills) — dev-time only, never a runtime dependency |

## Datasets (verify license before use — `/dataset-license`)
| Dataset | License (as found) | Track |
|---|---|---|
| Kaggle ISLR `asl-signs` (94,477 seqs, 250 signs, 21 signers) | Rules "Data Access and Use": any purpose incl. commercial + CC-By 4.0 (verified 2026-10-04; attribution required; see `docs/datasets.md`) | Release |
| Own Studio recordings | Consent form, CC BY 4.0 | Release |
| ASL Citizen | Microsoft research license, non-commercial | Research |
| Sem-Lex | Unverified (assume NC) | Research |
| WLASL, How2Sign | Non-commercial | Research |

## Lexicon (vocabulary DB)
- Own core lexicon, authored and reviewed in Signer Studio, licensed CC BY 4.0, shipped as versioned bundles (`YYYY.MM.N`).
- ASL Signbank (CC BY-NC-SA 4.0): ID-gloss conventions and external links only.
- ASL-LEX 2.0 (CC BY-NC 4.0): phonology coding used as our field schema; data only in an optional reference pack, never bundled.
- Review: edit = 1 reviewer approval; new sign = 2, at least one Deaf reviewer.
- New sign recognizable via prototypes when ≥10 examples from ≥3 signers.

## Interfaces (keep stable; schemas in `packages/schemas`)
- Landmark frame: float32 `[T, N, 3]`, layout `slk-landmarks-v1` (N=146: hands 2×21 + upper-body pose subset + face subset, ADR-0004), NaN = missing, centered on neck, scaled by shoulder width (ADR-0005), coordinates in `reference_aspect` 0.78 (ADR-0007), hands assigned to the nearer pose wrist (ADR-0008)
- Recognition output: `{gloss, gloss_id, confidence, t_start, t_end, top_k, model_version, lexicon_version}`
- WebSocket: `{"type": "caption"|"gloss"|"sign_seq", "session": id, "payload": {...}}`
- Lexicon entry: see PLAN §4.4
- Sign output request: `{"lang": "ase", "glosses": [str], "source_text": str}`
- `CaptionSink` and `SignRenderer` TS interfaces (PLAN §4.6–4.7)

## Evaluation protocol
- Signer-independent splits only; extra held-out set of own recorded signers
- Track: top-1/top-5, subgroup gaps, false activations, gloss WER, chrF, latency p50/p95, fps on benchmark laptop
- Review gates G0–G5 by fluent/Deaf signers (PLAN §9)

## Decision log
- pre-2026-10-01 — Core pipeline has no LLM/API dependency; LLM optional post-processor only
- pre-2026-10-01 — Recognition runs on landmarks, not pixels
- pre-2026-10-01 — Gloss layer is the shared intermediate representation
- 2026-10-01 — ASL ↔ English only; BdSL dropped
- 2026-10-01 — Apache-2.0 code; two data tracks; release models use permissive data only (CI-enforced)
- 2026-10-01 — Delivery: web demo → PWA → Tauri; meeting integration via overlay/virtual cam/local audio/caption APIs; no bots
- 2026-10-01 — Lexicon: own CC BY 4.0 core via Signer Studio; ASL-LEX/Signbank as references only
- 2026-10-01 — Recognizer = encoder + classifier + prototypes (Studio signs without retraining)
- 2026-10-04 — Repo at `C:\dev\signlnk` (outside OneDrive) on GitHub; one branch + PR per Phase step
- 2026-10-04 — `slk-landmarks-v1` N=146, Holistic→Tasks identity mapping (ADR-0004, ADR-0006)
- 2026-10-05 — Landmarks expressed in `reference_aspect=0.78`, calibrated on 7 recordings from one webcam (ADR-0007)
- 2026-10-05 — Hands assigned to the nearer pose wrist instead of MediaPipe's handedness label (wrong in ~7% of single-hand frames) (ADR-0008; supersedes ADR-0004's labels-as-is)
- 2026-10-05 — Claude Code framework adopted: committed `.claude/` (settings, 5 hooks, 6 path-scoped rules, 4 subagents, 7 skills), procedures in `docs/WORKFLOW.md`, history in `docs/CHANGELOG.md`; repo is canonical, claude.ai project docs are mirrors

## Open questions / risks
- `reference_aspect=0.78` calibrated on one webcam; one fresh clip from a second camera passes parity (2026-10-06). More cameras still welcome, not blocking
- ADR-0001/0002/0003 (Phase 0 step 7) not yet written
- Holistic↔Tasks face-index identity `[VERIFY]` (ADR-0004); hand side now follows the pose wrist (ADR-0008)
- numpy pinned to 2.4.6 (2.5.x needs Python ≥3.12; project keeps 3.11+)
- Kaggle vocab may lack meeting signs — fill via Studio
- No Deaf advisors yet — blocks public demo (G1)
- Claude Code features used by the framework need recent versions (verify-before-commit v2.1.286, `/skill-doctor` v2.1.252) — `[VERIFY]` with `claude --version`
- Facial grammar coverage; avatar intelligibility; continuous segmentation

---

## Appendix — Model and resource routing
Opus 5.5: architecture, hard training/debugging, translation design, phase-boundary reviews (`/phase-gate`, `plan-reviewer`).
Sonnet 5.5: day-to-day coding, components, scripts, explanations (`privacy-reviewer`, `license-auditor`).
Haiku 4.5: lookups, small refactors, test running (`test-runner`).
Claude Code on Windows: multi-file implementation, tests, repo management (`docs/WORKFLOW.md` §1).
Compute: landmarks + inference on CPU/browser; training on Kaggle/Colab free tiers.
After each session: `/session-close` updates this file; re-upload changed docs to the claude.ai project.
