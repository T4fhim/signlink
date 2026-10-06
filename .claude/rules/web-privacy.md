---
paths:
  - "apps/web/**"
  - "packages/landmarks/src/worker.ts"
  - "packages/landmarks/src/pipeline.ts"
  - "packages/landmarks/src/recorder.ts"
  - "packages/landmarks/src/protocol.ts"
  - "scripts/fetch-mediapipe-assets.mjs"
  - "scripts/mediapipe-assets.json"
---
<!-- Loads when Claude works on the web app, capture pipeline or asset fetching. -->
# Web, capture and privacy rules (CLAUDE.md rule 2, PLAN §10)

- Frames, MediaStreams, audio and landmark tensors never leave the device: no fetch/WebSocket/beacon
  carrying them, no uploads. Only model, WASM and lexicon bundle downloads are allowed.
- No third-party scripts, CDNs, fonts or analytics. MediaPipe WASM and models are self-hosted under
  `apps/web/public/mediapipe/` via `pnpm assets`.
- Capture shows a visible indicator; nothing records by default; downloads are user-initiated.
- Before touching `apps/web`, read `apps/web/AGENTS.md`: this Next.js version differs from your
  training data.
- UI for Deaf users must be accessible (keyboard, contrast, captions readable) and is reviewed at the
  community gates before it is called correct.
- After changes here, ask the `privacy-reviewer` subagent to check the diff.
