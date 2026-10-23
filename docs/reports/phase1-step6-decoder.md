# Phase 1 step 6: live decoder (code built, Done-when not yet measured)

Branch `feat/phase1-live-decoder`, 2026-10-24. PLAN §7 Phase 1 step 6 **Done when:** p95 sign-end →
caption < 500 ms **and** ≥ 25 fps with inference running, on the benchmark laptop.

**Status: the acceptance numbers do not exist yet.** Measuring them needs a person signing on camera
on the benchmark laptop (i5-1135G7, per `PROJECT_CONTEXT.md`). Everything below was measured on a
different machine without a camera and does **not** count toward the Done-when.

## What was built

| Piece | Where |
|---|---|
| TS `make_input` + `resample_nearest`, parity with Python | `packages/landmarks/src/modelInput.ts`; fixture `tests/fixtures/model_input/cases.json` (`uv run python -m signlnk_ml.training.golden_fixtures`) |
| Endpointer (hand-speed segmentation), decoder, `model.json` parser | `packages/recognizer/src/decoder/` |
| Inference worker (`onnxruntime-web/wasm`), host logic, client | `packages/recognizer/src/{runtime,protocol,host,inferenceWorker,client}.ts` |
| Measurement harness (whole-run fps, latency split) | `packages/recognizer/src/stats.ts` |
| Pages | `/dev/decode` (camera, captions, report download), `/dev/parity` (browser parity) |
| Assets | `scripts/fetch-ort-assets.mjs` (via `pnpm assets` / `predev`), `scripts/stage-model.mjs` |

## Measured on this machine (not the benchmark laptop)

Machine: AMD Ryzen 7 7840H, 8 cores / 16 logical (`Get-CimInstance Win32_Processor`), Chrome 155,
model `baseline-seed1` (251 classes, release track, Kaggle ISLR), served by `pnpm dev`.

- **Browser parity** (`/dev/parity`, 8 saved windows through the worker, `onnxruntime-web/wasm`):
  max abs logits diff **3.58e-6**, embeddings **2.03e-6**, tolerance 1e-4: PASS. Identical with 1
  and 4 threads. All page requests same-origin; the only 404 was `favicon.ico`. Output saved in
  `tests/decoder/browser-parity.json` (the same windows as the Node check of step 5, so the same
  difference is expected).
- **One window through the model**, 30 runs after one warm-up (PLAN §2.2 budget ≤ 40 ms):

  | | threads | p50 | p95 |
  |---|---|---|---|
  | A: default headers | 1 | 9.6 ms | 11.1 ms |
  | B: `SIGNLNK_ISOLATE=1` (COOP/COEP) | 4 | 4.75 ms | 6.97 ms |

  Both are far inside the budget, so threads are not needed; isolation stays an opt-in flag.
- `/dev/decode` loads the model, worker and wasm, then waits at the camera permission (this
  automated browser has no camera). Decoding on real signing is untested.
- Node: `pnpm test` landmarks 84, recognizer 66, schemas 17 passed; `uv run pytest` 203 passed;
  mypy, ruff, typecheck clean. The host test feeds a synthetic sign through `InferenceHost` with
  `tiny.onnx` and checks the caption equals an independent `Decoder` + runtime on the same frames.

## Not measured, and what blocks it

1. **fps with inference running** (whole-run mean ≥ 60 s, both hands signing): needs the benchmark
   laptop and a camera.
2. **p95 sign-end → caption** (`signEndToCaptionMs` in the report): same.
3. Endpointer thresholds (`startSpeed` 0.6, `stopSpeed` 0.3 shoulder widths/s, `stopMs` 200,
   preroll 150, tail 100) are untuned starting values `[VERIFY]`.
4. Whether real signing, segmented by the endpointer, scores near the training-clip accuracy: the
   model was trained on whole Kaggle clips, the live input is an endpointed segment.

5. **Automated same-origin check.** `docs/training.md` ("Using the model in the browser") asks for a
   Playwright test that every request in a recognition run is same-origin. There is no Playwright
   dependency yet, so it is **deferred** (adding one is a new dependency and a CI change; decide
   with step 7's offline test). Evidence so far is manual: the DevTools request list of a
   `/dev/parity` run (model, worker chunks, ORT `.mjs` and `.wasm`, including the worker's own
   fetches) held only `localhost:3000` URLs.

How the report counts latency: a sign cut at `maxSignMs` while the hands still move has no real end
(its `tEnd` is about the moment of the cut), so such signs are counted (`signs.forced`) but left out
of `signEndToCaptionMs` and `decoderWaitMs`. The latency target counts as met only with at least 20
captioned signs, because below that a nearest-rank p95 is just the maximum.

How to measure on the benchmark laptop: `pnpm install`, `pnpm assets`,
`node scripts/stage-model.mjs <run-dir> --vocab ml/vocab/draft-meeting-wishlist.yaml --size 20`
(`--size` keeps the first N signs, so runs match the 20-sign proxy), `pnpm dev`, open `/dev/decode`, start, sign for ≥ 60 s with both hands, download the report (suggested home:
`tests/decoder/decode-benchmark.json`, like `tests/landmarks/`).

## Findings along the way

- Under COEP `require-corp`, Chrome blocks worker scripts without the header, including ORT's own
  thread workers in `/ort`. The isolation rule therefore covers `/dev`, `/_next` and `/ort`.
- A blocked worker script fires `error` with an empty message; the first version showed nothing.
  `client.ts` now reports "the inference worker failed to start".
- `PROJECT_CONTEXT.md` names the i5-1135G7 as "this machine"; this machine is a Ryzen 7 7840H.
  Update the file once it is clear which machine is the benchmark laptop.

## Decisions to record (run `/adr`; ADR-0009 proposed)

0. **PLAN wording.** PLAN §7 says "sliding window"; the code endpoints a sign and resamples that
   segment instead (item 1). When the ADR is written, update the step's text to match, and
   `PROJECT_CONTEXT.md` (decision log, Current phase) through `/session-close`.
1. **Endpoint, then resample.** Training windows are whole clips resampled to 64 frames, so the live
   decoder cuts a segment (hand speed opens and closes it) and resamples that with the same
   nearest-frame formula. A fixed 64-raw-frame slide is a different input distribution.
2. **Vocabulary mask at inference.** Softmax over the vocabulary plus `<other>`; the best sign must
   beat `<other>` and `minConfidence`. Supersedes the "proposed" note of 2026-10-19.
3. **Temporary gloss mapping.** `gloss` = class name upper-cased, `gloss_id` = `<language>:<GLOSS>`,
   `lexicon_version` = `0000.0.0` until a lexicon bundle exists. No schema change. The language
   (`ase`) and these placeholders live in one module, `apps/web/app/dev/decoderIdentity.ts`;
   `model.json` records no language yet, so a real config source is still to be chosen (rule 3).
4. The decoder's share of the latency budget is the endpointer's `stopMs`; the sign end is the last
   moving frame, not the frame that closed the segment.

## Known limits

- Velocity models are rejected (`parseModelInfo`); only the shipped `velocity: false` input is ported.
- The default `onnxruntime-web` browser entry (jsep/WebGPU) is not used or tested.
- The same gloss within `debounceMs` of the previous one is dropped, so a deliberate quick repeat is lost.
- ONNX Runtime's LICENSE / ThirdPartyNotices must ship with the wasm (step 7).
