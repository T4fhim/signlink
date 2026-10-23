import { LANDMARK_COUNT, shoulderIndices } from "@signlnk/landmarks";
import { describe, expect, it } from "vitest";
import { Decoder, type DecoderConfig, type PendingSign } from "./decoder.ts";
import type { ModelInfo } from "./modelInfo.ts";

const N = LANDMARK_COUNT;
const model: ModelInfo = {
  classes: ["hello", "thankyou", "yes", "no", "<other>"],
  otherIndex: 4,
  length: 64,
  dim: 438,
  useZ: false,
  nLandmarks: N,
};
const config = (patch: Partial<DecoderConfig> = {}): DecoderConfig => ({
  model,
  language: "ase",
  modelVersion: "isr-0.0.0",
  lexiconVersion: "0000.0.0",
  vocabulary: null,
  minConfidence: 0.6,
  debounceMs: 600,
  topK: 3,
  ...patch,
});
const pending = (tStart = 1000, tEnd = 1800): PendingSign => ({
  input: new Float32Array(64 * 438),
  tStart,
  tEnd,
  tDetected: tEnd + 220,
  frameCount: 30,
  reason: "quiet",
});
const softmax = (values: number[]): number[] => {
  const top = Math.max(...values);
  const e = values.map((v) => Math.exp(v - top));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
};

describe("Decoder.interpret", () => {
  it("returns the best sign as a RecognitionOutputV1", () => {
    const logits = Float32Array.from([4, 1, 0, 0, -2]);
    const caption = new Decoder(config()).interpret(pending(1000.4, 1800.6), logits);
    const p = softmax([4, 1, 0, 0, -2]);
    expect(caption).toMatchObject({
      gloss: "HELLO",
      gloss_id: "ase:HELLO",
      t_start: 1000,
      t_end: 1801,
      model_version: "isr-0.0.0",
      lexicon_version: "0000.0.0",
    });
    expect(caption?.confidence).toBeCloseTo(p[0] as number, 5);
    expect(caption?.top_k.map((k) => k.gloss_id)).toEqual(["ase:HELLO", "ase:THANKYOU", "ase:YES"]);
    expect(caption?.top_k[0]?.p).toBeCloseTo(p[0] as number, 5);
    expect(caption?.gloss_id).toMatch(/^[a-z]{2,3}:.+$/);
    expect(caption?.lexicon_version).toMatch(/^[0-9]{4}\.[0-9]{1,2}\.[0-9]+$/);
  });

  it("restricts to the vocabulary and renormalizes over it plus <other>", () => {
    // "no" has the highest logit overall but is outside the vocabulary.
    const logits = Float32Array.from([2, 3, 0, 9, 0]);
    const caption = new Decoder(config({ vocabulary: ["hello", "thankyou"] })).interpret(
      pending(),
      logits,
    );
    const p = softmax([2, 3, 0]); // hello, thankyou, <other>
    expect(caption?.gloss).toBe("THANKYOU");
    expect(caption?.confidence).toBeCloseTo(p[1] as number, 5);
    expect(caption?.top_k).toHaveLength(2);
  });

  it("emits nothing when <other> beats every sign", () => {
    expect(
      new Decoder(config()).interpret(pending(), Float32Array.from([1, 1, 0, 0, 5])),
    ).toBeNull();
  });

  it("emits nothing below minConfidence", () => {
    const logits = Float32Array.from([1, 0.9, 0.8, 0.7, -9]);
    expect(new Decoder(config()).interpret(pending(), logits)).toBeNull();
  });

  it("suppresses the same gloss within debounceMs, then allows it again", () => {
    const decoder = new Decoder(config());
    const hello = Float32Array.from([6, 0, 0, 0, 0]);
    expect(decoder.interpret(pending(1000, 1800), hello)).not.toBeNull();
    expect(decoder.interpret(pending(2200, 2800), hello)).toBeNull(); // 400 ms after the last end
    expect(decoder.interpret(pending(4000, 4600), hello)).not.toBeNull(); // 1200 ms after it
  });

  it("does not debounce a different gloss", () => {
    const decoder = new Decoder(config());
    expect(
      decoder.interpret(pending(1000, 1800), Float32Array.from([6, 0, 0, 0, 0])),
    ).not.toBeNull();
    expect(
      decoder.interpret(pending(1900, 2400), Float32Array.from([0, 6, 0, 0, 0])),
    ).not.toBeNull();
  });

  it("rejects logits of the wrong length", () => {
    expect(() => new Decoder(config()).interpret(pending(), new Float32Array(4))).toThrow(
      RangeError,
    );
  });
});

describe("Decoder construction", () => {
  it.each([
    ["an unknown sign", ["hello", "nope"]],
    ["<other>", ["hello", "<other>"]],
    ["a repeated sign", ["hello", "hello"]],
  ])("rejects a vocabulary with %s", (_label, vocabulary) => {
    expect(() => new Decoder(config({ vocabulary }))).toThrow(/vocabulary/);
  });
});

describe("Decoder.push", () => {
  const { left, right } = shoulderIndices();
  /** A raw (not normalized) frame: shoulders 0.3 apart, right hand at x. */
  const raw = (handX: number | null): Float32Array => {
    const f = new Float32Array(N * 3).fill(Number.NaN);
    f.set([0.65, 0.5, 0], left * 3);
    f.set([0.35, 0.5, 0], right * 3);
    if (handX !== null) for (let i = 21; i < 42; i++) f.set([handX, 0.4, 0], i * 3);
    return f;
  };

  it("turns a signed movement into a model-ready window", () => {
    const decoder = new Decoder(
      config({ endpointer: { smoothing: 1, startMs: 60, stopMs: 200, minSignMs: 200 } }),
    );
    const dt = 1000 / 30;
    const found: PendingSign[] = [];
    let x = 0.4;
    for (let t = 0; t < 2400; t += dt) {
      if (t >= 500 && t < 1300) x += (0.45 * dt) / 1000; // 0.45 raw/s = 1.5 shoulder widths/s
      const hand = t < 300 ? null : x;
      const got = decoder.push(t, raw(hand));
      if (got) found.push(got);
    }
    expect(found).toHaveLength(1);
    const sign = found[0] as PendingSign;
    expect(sign.input.length).toBe(64 * 438);
    expect(Array.from(sign.input).some(Number.isNaN)).toBe(false);
    expect(sign.tStart).toBeGreaterThanOrEqual(500);
    expect(sign.tEnd).toBeLessThanOrEqual(1300);
    expect(sign.tDetected).toBeGreaterThan(sign.tEnd);
    // presence flags: the right hand is there, the left hand never is
    const presence = sign.input.subarray(N * 2, N * 3);
    expect(presence[25]).toBe(1);
    expect(presence[5]).toBe(0);
    expect(sign.reason).toBe("quiet");
  });

  it("marks a sign cut at maxSignMs while the hands still move", () => {
    const decoder = new Decoder(
      config({
        endpointer: { smoothing: 1, startMs: 60, stopMs: 200, minSignMs: 200, maxSignMs: 600 },
      }),
    );
    const dt = 1000 / 30;
    const found: PendingSign[] = [];
    let x = 0.4;
    for (let t = 0; t < 1500; t += dt) {
      x += (0.45 * dt) / 1000; // keeps moving the whole time
      const got = decoder.push(t, raw(x));
      if (got) found.push(got);
    }
    expect(found.length).toBeGreaterThanOrEqual(1);
    expect(found[0]?.reason).toBe("max");
  });
});
