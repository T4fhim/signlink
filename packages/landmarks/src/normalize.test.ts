import { readdirSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { LANDMARK_COUNT } from "./layout.ts";
import {
  MIN_SHOULDER_WIDTH,
  normalizeFrame,
  normalizeSequence,
  shoulderIndices,
} from "./normalize.ts";

const TOLERANCE = 1e-5; // max abs diff, the TS/Python parity budget
const N = LANDMARK_COUNT;
const FRAME = N * 3;
const { left: LEFT, right: RIGHT } = shoulderIndices();

const fixtureDir = new URL("../../../tests/fixtures/normalization/", import.meta.url);
const fixtureFiles = readdirSync(fixtureDir)
  .filter((f) => f.endsWith(".json"))
  .sort();

interface Fixture {
  name: string;
  layout_id: string;
  shape: [number, number, number];
  input: (number | null)[];
  expected: (number | null)[];
}
const toFloat32 = (values: (number | null)[]): Float32Array =>
  Float32Array.from(values, (v) => v ?? Number.NaN);

function expectClose(actual: Float32Array, expected: Float32Array, label = ""): void {
  expect(actual.length, label).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const a = actual[i] ?? Number.NaN;
    const e = expected[i] ?? Number.NaN;
    if (Number.isNaN(e)) {
      expect(Number.isNaN(a), `${label}[${i}] should be NaN`).toBe(true);
    } else {
      expect(Math.abs(a - e), `${label}[${i}]: ${a} vs ${e}`).toBeLessThanOrEqual(TOLERANCE);
    }
  }
}

/** Deterministic pseudo-random numbers in [0, 1). */
function mulberry32(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Fully finite landmarks with plausible, non-degenerate shoulders. */
function cleanFrames(frames = 5, seed = 7): Float32Array {
  const random = mulberry32(seed);
  const a = Float32Array.from({ length: frames * FRAME }, () => random());
  for (let t = 0; t < frames; t++) {
    const base = t * FRAME;
    a.set([0.65, 0.55], base + LEFT * 3);
    a.set([0.35, 0.6], base + RIGHT * 3);
  }
  return a;
}

describe("normalization parity with the Python reference", () => {
  it("has the three golden fixtures", () => {
    expect(fixtureFiles).toEqual(["edge_cases.json", "hand_computed.json", "random_clip.json"]);
  });

  it.each(fixtureFiles)("matches golden fixture %s", (file) => {
    const fixture = JSON.parse(readFileSync(new URL(file, fixtureDir), "utf8")) as Fixture;
    expect(fixture.layout_id).toBe("slk-landmarks-v1");
    expect(fixture.shape.slice(1)).toEqual([N, 3]);
    const actual = normalizeSequence(toFloat32(fixture.input));
    expectClose(actual, toFloat32(fixture.expected), file);
  });
});

describe("normalizeFrame", () => {
  it("normalizes a hand-computed frame (centre 0.5/0.5, width 0.2, z kept)", () => {
    const a = new Float32Array(FRAME).fill(Number.NaN);
    a.set([0.6, 0.5, 0.0], LEFT * 3);
    a.set([0.4, 0.5, 0.0], RIGHT * 3);
    a.set([0.5, 0.3, 0.7], 0);
    a.set([0.7, 0.9, -0.4], 3);
    a.set([0.4, 0.5, 0.25], 100 * 3);

    const expected = new Float32Array(FRAME).fill(Number.NaN);
    expected.set([0.5, 0, 0], LEFT * 3);
    expected.set([-0.5, 0, 0], RIGHT * 3);
    expected.set([0, -1, 0.7], 0);
    expected.set([1, 2, -0.4], 3);
    expected.set([-0.5, 0, 0.25], 100 * 3);
    expectClose(normalizeFrame(a), expected);
  });

  it("is translation invariant", () => {
    const a = cleanFrames(1);
    const shifted = a.slice();
    for (let i = 0; i < FRAME; i += 3) {
      shifted[i] = (shifted[i] ?? 0) + 0.13;
      shifted[i + 1] = (shifted[i + 1] ?? 0) - 0.21;
    }
    expectClose(normalizeFrame(shifted), normalizeFrame(a));
  });

  it("is uniform-scale invariant", () => {
    const a = cleanFrames(1);
    const scaled = a.slice();
    for (let i = 0; i < FRAME; i += 3) {
      scaled[i] = (scaled[i] ?? 0) * 2.5;
      scaled[i + 1] = (scaled[i + 1] ?? 0) * 2.5;
    }
    expectClose(normalizeFrame(scaled), normalizeFrame(a));
  });

  it("keeps z unchanged", () => {
    const a = cleanFrames(1);
    const out = normalizeFrame(a);
    for (let i = 2; i < FRAME; i += 3) expect(out[i]).toBe(a[i]);
  });

  it("puts the shoulders one unit apart around the origin", () => {
    const out = normalizeFrame(cleanFrames(1));
    const [lx = NaN, ly = NaN] = out.subarray(LEFT * 3, LEFT * 3 + 2);
    const [rx = NaN, ry = NaN] = out.subarray(RIGHT * 3, RIGHT * 3 + 2);
    expect(Math.abs((lx + rx) / 2)).toBeLessThan(TOLERANCE);
    expect(Math.abs((ly + ry) / 2)).toBeLessThan(TOLERANCE);
    expect(Math.abs(Math.hypot(lx - rx, ly - ry) - 1)).toBeLessThan(TOLERANCE);
  });

  it("keeps missing landmarks missing and leaves the rest finite", () => {
    const a = cleanFrames(1);
    a.set([Number.NaN, Number.NaN, Number.NaN], 3 * 3);
    a[5 * 3 + 1] = Number.NaN;
    const out = normalizeFrame(a);
    expect(Array.from(out.subarray(9, 12)).every(Number.isNaN)).toBe(true);
    expect(Number.isNaN(out[5 * 3 + 1])).toBe(true);
    expect(Number.isFinite(out[5 * 3])).toBe(true);
    expect(Number.isFinite(out[5 * 3 + 2])).toBe(true);
    expect(Number.isFinite(out[0])).toBe(true);
  });

  it.each([LEFT, RIGHT])("turns a frame without shoulder %i into all NaN", (anchor) => {
    const a = cleanFrames(1);
    a[anchor * 3] = Number.NaN;
    expect(Array.from(normalizeFrame(a)).every(Number.isNaN)).toBe(true);
  });

  it("treats shoulder widths below MIN_SHOULDER_WIDTH as degenerate", () => {
    const make = (extra: number): Float32Array => {
      const a = cleanFrames(1);
      a.set(a.subarray(LEFT * 3, LEFT * 3 + 2), RIGHT * 3);
      a[RIGHT * 3] = (a[RIGHT * 3] ?? 0) + extra;
      return a;
    };
    expect(Array.from(normalizeFrame(make(MIN_SHOULDER_WIDTH * 0.5))).every(Number.isNaN)).toBe(
      true,
    );
    expect(Array.from(normalizeFrame(make(MIN_SHOULDER_WIDTH * 4))).every(Number.isFinite)).toBe(
      true,
    );
  });

  it("does not modify its input", () => {
    const a = cleanFrames(1);
    const before = a.slice();
    normalizeFrame(a);
    expect(a).toEqual(before);
  });

  it("rejects a frame of the wrong length", () => {
    expect(() => normalizeFrame(new Float32Array(FRAME - 1))).toThrow(RangeError);
  });
});

describe("normalizeSequence", () => {
  it("equals normalizing each frame on its own", () => {
    const clip = cleanFrames(3);
    const out = normalizeSequence(clip);
    for (let t = 0; t < 3; t++) {
      const one = normalizeFrame(clip.slice(t * FRAME, (t + 1) * FRAME));
      expect(Array.from(out.subarray(t * FRAME, (t + 1) * FRAME))).toEqual(Array.from(one));
    }
  });

  it("rejects a length that is not a whole number of frames", () => {
    expect(() => normalizeSequence(new Float32Array(FRAME + 1))).toThrow(RangeError);
  });
});
