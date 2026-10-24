# Phase 1 step 6: live decoder (fps met on the home PC, latency sample too small, accuracy poor)

Branch `feat/phase1-live-decoder`, 2026-10-24. PLAN §7 Phase 1 step 6 **Done when:** p95 sign-end →
caption < 500 ms **and** ≥ 25 fps with inference running, on the benchmark laptop.

**Benchmark PC.** It is chosen per day (CLAUDE.md step 0). For this run it was the **home PC, AMD
Ryzen 7 7840H** (the run reports 16 logical cores). Results from another PC need their own run.

**Status:** fps target met on this PC; the latency target is not yet shown (13 of the 20 samples it
needs); live recognition on real signing was visibly inaccurate (see below). Numbers in the next
section come from `D:\signlnk-data\serve-recordings\decode-benchmark.json` (not in the repo; the
suggested home is `tests/decoder/decode-benchmark.json`).

## Live run on the home PC (Ryzen 7 7840H, Chrome 155, 2026-10-23, camera, signing)

Options: GPU delegate, `poseEvery` 2, `faceEvery` 2, `vocabulary` all (251 classes, no mask),
`minConfidence` 0.5, `debounceMs` 600, endpointer defaults (`stopMs` 200), 1 ORT thread, not
cross-origin isolated. Checked against the file: fps = (8,790 − 1) frames / 208.45 s = 42.16; counts
reconcile (47 signs, 6 forced, 41 decoder waits, 13 painted at rest).

| | Result | Target / note |
|---|---|---|
| Whole-run fps | **42.16** (208 s, 8,790 frames, longest gap 202 ms) | ≥ 25 over ≥ 60 s: **met** |
| Hands in view | either 44.9%, right 44.8%, left 3.7%, both 3.5% | run was mostly not signing |
| Landmark ms | p50 24.1, p95 39.6, max 5,931 | max is probably cold start `[VERIFY]` |
| Model run | p50 5.5 ms, p95 6.3 | |
| Sign end → caption | p50 256 ms, p95 319.5 (= max), n = 13 | < 500 ms, but needs n ≥ 20: **not met** |
| Of which endpointer wait / pipeline | 212 / 23.7 ms p50 | wait ≈ `stopMs` 200 |
| Signs | 47 segments, 15 captioned, 6 forced at `maxSignMs` | |

Caveats on the fps: it is a whole-run mean with hands in view in under half the frames, pose and
face only every second frame, and the GPU delegate; the rate during sustained two-hand signing is
not separated out. Landmark p95 of 39.6 ms is about 25 fps for the slowest frames.

**Accuracy observed by the signer (not measured, no per-sign log):** "see you later" yielded only
"see"; pointing at self for "I" yielded "thirsty". Known reasons: the 251-class model has **no
"I", "me" or "you" class** (it has "see", "later", "thirsty"), so those signs can only be wrong or
`<other>`; the run used no vocabulary mask; a phrase signed without pauses may be one segment for
the endpointer. Why "later" was lost (merged segment, confidence under 0.5, or debounce) cannot be
told from the report. A per-sign log (segment length, top-3, confidence) would answer it; the
signer has ideas for improving the learning, to follow.

## What was built

| Piece | Where |
|---|---|
| TS `make_input` + `resample_nearest`, parity with Python | `packages/landmarks/src/modelInput.ts`; fixture `tests/fixtures/model_input/cases.json` (`uv run python -m signlnk_ml.training.golden_fixtures`) |
| Endpointer (hand-speed segmentation), decoder, `model.json` parser | `packages/recognizer/src/decoder/` |
| Inference worker (`onnxruntime-web/wasm`), host logic, client | `packages/recognizer/src/{runtime,protocol,host,inferenceWorker,client}.ts` |
| Measurement harness (whole-run fps, latency split) | `packages/recognizer/src/stats.ts` |
| Pages | `/dev/decode` (camera, captions, report download), `/dev/parity` (browser parity) |
| Assets | `scripts/fetch-ort-assets.mjs` (via `pnpm assets` / `predev`), `scripts/stage-model.mjs` |

## Parity, single-window latency and tests (same PC, no camera)

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
  automated browser has no camera). The live run with a camera is in the section above.
- Node: `pnpm test` landmarks 84, recognizer 66, schemas 17 passed; `uv run pytest` 203 passed;
  mypy, ruff, typecheck clean. The host test feeds a synthetic sign through `InferenceHost` with
  `tiny.onnx` and checks the caption equals an independent `Decoder` + runtime on the same frames.

## Not measured, and what blocks it

1. **fps while signing with both hands**: the home-PC run met 25 fps as a whole-run mean, but hands
   were in view in 45% of frames and both in 3.5%. Repeat with sustained two-hand signing, and on
   the other PC on a day it is the benchmark.
2. **p95 sign-end → caption** (`signEndToCaptionMs`): 319.5 ms from 13 samples, below the 20 the
   report needs to call it met. Needs a longer run with at least 20 captioned signs that end at rest.
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

How to measure on the day's benchmark PC: `pnpm install`, `pnpm assets`,
`node scripts/stage-model.mjs <run-dir> --vocab ml/vocab/draft-meeting-wishlist.yaml --size 20`
(`--size` keeps the first N signs, so runs match the 20-sign proxy), `pnpm dev`, open `/dev/decode`, start, sign for ≥ 60 s with both hands, download the report (suggested home:
`tests/decoder/decode-benchmark.json`, like `tests/landmarks/`).

## Findings along the way

- Under COEP `require-corp`, Chrome blocks worker scripts without the header, including ORT's own
  thread workers in `/ort`. The isolation rule therefore covers `/dev`, `/_next` and `/ort`.
- A blocked worker script fires `error` with an empty message; the first version showed nothing.
  `client.ts` now reports "the inference worker failed to start".
- `PROJECT_CONTEXT.md` named the i5-1135G7 as "this machine"; the runs above are on a Ryzen 7 7840H.
  Resolved 2026-10-24: the benchmark PC is asked each day (home Ryzen or work PC), and PROJECT_CONTEXT
  and CLAUDE.md say so. The work PC's spec and the i5's current role are still `[VERIFY]`.

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
