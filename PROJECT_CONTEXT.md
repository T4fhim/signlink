# SignLnk (working name) — Project Context
Last updated: 2026-10-04
Long-form plan: `docs/PLAN.md` (architecture, interfaces, Studio design, step-by-step build)
Toolkit: `docs/TOOLKIT.md` (agents, skills, MCPs by phase status)

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

## Current phase
**Phase 0 — Setup** (not yet built). Next: Phase 1 — isolated signs + Studio v1.

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

## Datasets (verify license before use)
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
- Landmark frame: float32 `[T, N, 3]`, layout `slk-landmarks-v1` (hands 2×21 + upper-body pose subset + face subset), NaN = missing, centered on neck, scaled by shoulder width
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

## Open questions / risks
- Train/serve landmark mismatch (legacy Holistic vs Tasks face points) — parity test in Phase 0
- Kaggle vocab may lack meeting signs — fill via Studio
- No Deaf advisors yet — blocks public demo
- numpy pinned to 2.4.6 (2.5.x needs Python ≥3.12; project keeps 3.11+)
- Left/right hand agreement with the Holistic-extracted Kaggle data and Holistic↔Tasks face-index identity are both `[VERIFY]` in step 5 (ADR-0004)
- Facial grammar coverage; avatar intelligibility; continuous segmentation

## Changelog
- 2026-10-04 — Phase 0 step 4 merged (PR #4). Step 5 built (branch `feat/phase0-islr`): Kaggle ISLR loader (Holistic → slk-landmarks-v1 via `legacy_holistic_indices`, identity), geometry checks, parity stats; 525-sequence sample (21 signers, 311 MB) in `D:\signlnk-data`; Kaggle licence verified (ADR-0006, `docs/datasets.md`). Serve-side parity waits for a `/dev/record` recording (step 6).
- 2026-10-04 — Phase 0 step 3 merged (PR #3). Step 4 built (branch `feat/phase0-normalization`): neck-centred, shoulder-width-scaled normalization in TS and Python, z kept, invalid frames all NaN (ADR-0005); parity ≤1e-5 on 3 golden fixtures, mutation-checked.
- 2026-10-04 — Phase 0 steps 1–2 merged (PR #1, #2): monorepo, CI, five v1 schemas, TS+Pydantic generation with staleness check. Repo now at `C:\dev\signlnk` (outside OneDrive).
- 2026-10-04 — Phase 0 step 3 built (branch `feat/phase0-landmarks`): `slk-landmarks-v1` N=146 (ADR-0004), MediaPipe worker pipeline, `/dev/landmarks` benchmark page. fps gate passed with a real signer: 35.1 fps, worker p95 56.6 ms (pose+face every 2nd frame). Handedness labels used as-is (a swap was wrong).
- 2026-10-01 — Decisions filled; PLAN.md v1 and CLAUDE.md written; phase corrected to 0.
- 2026-10-04 — Toolkit audited; `docs/TOOLKIT.md` added and linked from CLAUDE.md; ecc + engineering plugins active, qodo optional, miro off, no connectors needed.

---

## Appendix — Model and resource routing
Opus 5.5: architecture, hard training/debugging, translation design, phase-boundary reviews.
Sonnet 5.5: day-to-day coding, components, scripts, explanations. Haiku 4.5: lookups, small refactors.
Claude Code: multi-file implementation, tests, repo management (see `CLAUDE.md`).
Compute: landmarks + inference on CPU/browser; training on Kaggle/Colab free tiers.
After each session: paste "Log to context:" lines here, update date and phase, re-upload.
