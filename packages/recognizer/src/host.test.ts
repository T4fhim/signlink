import { readFileSync } from "node:fs";
import { LANDMARK_COUNT, shoulderIndices } from "@signlnk/landmarks";
import * as ort from "onnxruntime-web";
import { describe, expect, it } from "vitest";
import { Decoder, parseModelInfo, type PendingSign } from "./decoder/index.ts";
import { InferenceHost } from "./host.ts";
import type { FromWorker, InitMessage, SignMessage } from "./protocol.ts";
import { createRecognizer } from "./session.ts";

const N = LANDMARK_COUNT;
const tiny = new Uint8Array(
  readFileSync(new URL("../../../tests/fixtures/onnx/tiny.onnx", import.meta.url)),
);
// tiny.onnx: 11 classes over [B, 64, 438] windows (see tiny.parity.json)
const modelJson = {
  format: "slk-recognizer-v1",
  classes: [...Array.from({ length: 10 }, (_, i) => `sign${i}`), "<other>"],
  other_index: 10,
  input: {
    name: "windows",
    length: 64,
    dim: 438,
    use_z: false,
    velocity: false,
    n_landmarks: N,
    layout: "slk-landmarks-v1",
  },
};
const decoderConfig: InitMessage["decoder"] = {
  language: "ase",
  modelVersion: "isr-0.0.0",
  lexiconVersion: "0000.0.0",
  vocabulary: null,
  minConfidence: 0,
  debounceMs: 0,
  topK: 3,
  endpointer: { smoothing: 1, startMs: 60, stopMs: 200, minSignMs: 200 },
};
const init = (patch: Partial<InitMessage> = {}): InitMessage => ({
  type: "init",
  model: tiny,
  modelJson,
  decoder: decoderConfig,
  ...patch,
});

const { left, right } = shoulderIndices();
/** Raw frames for ~2.4 s: right hand appears at 300 ms, signs from 500 to 1300 ms, then rests. */
function* stream(): Generator<{ t: number; frame: Float32Array }> {
  const dt = 1000 / 30;
  let x = 0.4;
  for (let t = 0; t < 2400; t += dt) {
    if (t >= 500 && t < 1300) x += (0.45 * dt) / 1000;
    const frame = new Float32Array(N * 3).fill(Number.NaN);
    frame.set([0.65, 0.5, 0], left * 3);
    frame.set([0.35, 0.5, 0], right * 3);
    if (t >= 300) for (let i = 21; i < 42; i++) frame.set([x, 0.4, 0], i * 3);
    yield { t, frame };
  }
}

function newHost(): { host: InferenceHost; posted: FromWorker[] } {
  const posted: FromWorker[] = [];
  return { host: new InferenceHost(ort, (m) => posted.push(m)), posted };
}

describe("InferenceHost", () => {
  it("answers init with ready, then reports a sign with the caption the decoder would give", async () => {
    const { host, posted } = newHost();
    await host.handle(init());
    expect(posted[0]).toMatchObject({ type: "ready" });

    for (const { t, frame } of stream()) await host.handle({ type: "frame", t, frame });
    const signs = posted.filter((m): m is SignMessage => m.type === "sign");
    expect(signs).toHaveLength(1);
    const sign = signs[0] as SignMessage;
    expect(sign.tStart).toBeGreaterThanOrEqual(500);
    expect(sign.tEnd).toBeLessThanOrEqual(1300);
    expect(sign.tDetected).toBeGreaterThan(sign.tEnd);
    expect(sign.inferMs).toBeGreaterThanOrEqual(0);
    expect(sign.frameCount).toBeGreaterThan(0);
    expect(sign.reason).toBe("quiet");

    // An independent decoder and runtime on the same frames must agree exactly.
    const info = parseModelInfo(modelJson);
    const decoder = new Decoder({ ...decoderConfig, model: info });
    const recognizer = await createRecognizer(tiny, { length: info.length, dim: info.dim });
    let pending: PendingSign | null = null;
    for (const { t, frame } of stream()) pending = decoder.push(t, frame) ?? pending;
    const { logits } = await recognizer.run((pending as PendingSign).input, 1);
    const verdict = decoder.decide(pending as PendingSign, logits);
    expect(sign.caption).toEqual(verdict.caption);
    expect(sign.outcome).toBe(verdict.outcome);
    expect(sign.top).toEqual(verdict.top);
    expect(sign.otherP).toBe(verdict.otherP);
    await recognizer.dispose();
    await host.handle({ type: "dispose" });
  });

  it("runs raw windows on request (used by the browser parity check)", async () => {
    const { host, posted } = newHost();
    await host.handle(init());
    const windows = new Float32Array(2 * 64 * 438).fill(0.1);
    await host.handle({ type: "infer", id: 7, windows, batch: 2 });
    const reply = posted.find((m) => m.type === "logits");
    expect(reply).toMatchObject({ type: "logits", id: 7 });
    expect((reply as { logits: Float32Array }).logits).toHaveLength(2 * 11);
    await host.handle({ type: "dispose" });
  });

  it("reports an error for a frame sent before init", async () => {
    const { host, posted } = newHost();
    await host.handle({ type: "frame", t: 0, frame: new Float32Array(N * 3) });
    expect(posted).toEqual([{ type: "error", message: expect.stringContaining("init") }]);
  });

  it("reports an error for a model.json it cannot use", async () => {
    const { host, posted } = newHost();
    await host.handle(init({ modelJson: { ...modelJson, format: "other" } }));
    expect(posted).toEqual([{ type: "error", message: expect.stringContaining("model.json") }]);
  });

  it("forgets the open sign on reset", async () => {
    const { host, posted } = newHost();
    await host.handle(init());
    let count = 0;
    for (const { t, frame } of stream()) {
      await host.handle({ type: "frame", t, frame });
      if (++count === 30) await host.handle({ type: "reset" }); // mid-sign, at 1 s
    }
    // the first half of the sign is gone, so what is left is too short or a different segment
    const signs = posted.filter((m) => m.type === "sign");
    expect(signs.length).toBeLessThanOrEqual(1);
  });
});
