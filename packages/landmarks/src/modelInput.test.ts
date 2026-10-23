import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { LANDMARK_COUNT } from "./layout.ts";
import { inputDim, makeInput, resampleNearest } from "./modelInput.ts";

const TOLERANCE = 1e-5; // max abs diff, the TS/Python parity budget
const N = LANDMARK_COUNT;

interface MakeInputCase {
  name: string;
  use_z: boolean;
  shape: [number, number, number];
  input: (number | null)[];
  expected: (number | null)[];
}
interface ResampleCase {
  name: string;
  length: number;
  shape: [number, number, number];
  input: (number | null)[];
  expected: (number | null)[];
}
const fixture = JSON.parse(
  readFileSync(new URL("../../../tests/fixtures/model_input/cases.json", import.meta.url), "utf8"),
) as { make_input: MakeInputCase[]; resample: ResampleCase[] };

const toFloat32 = (values: (number | null)[]): Float32Array =>
  Float32Array.from(values, (v) => v ?? Number.NaN);

function expectClose(actual: Float32Array, expected: Float32Array, label: string): void {
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

describe("model input parity with the Python reference", () => {
  it.each(fixture.make_input.map((c) => [c.name, c] as const))("makeInput matches %s", (_n, c) => {
    expect(c.shape.slice(1)).toEqual([N, 3]);
    const out = makeInput(toFloat32(c.input), { useZ: c.use_z });
    expectClose(out, toFloat32(c.expected), c.name);
  });

  it.each(fixture.resample.map((c) => [c.name, c] as const))(
    "resampleNearest matches %s",
    (_n, c) => {
      const frameLength = c.shape[1] * c.shape[2];
      const out = resampleNearest(toFloat32(c.input), frameLength, c.length);
      expectClose(out, toFloat32(c.expected), c.name);
    },
  );
});

describe("inputDim", () => {
  it("is coordinates plus one presence flag per landmark", () => {
    expect(inputDim(N, false)).toBe(N * 3); // the shipped model: 146 x 3 = 438
    expect(inputDim(N, true)).toBe(N * 4);
  });
});

describe("makeInput", () => {
  const frame = (fill: number): Float32Array => new Float32Array(N * 3).fill(fill);

  it("maps NaN to 0 and flags only fully present landmarks", () => {
    const f = frame(0.5);
    f.set([Number.NaN, 0.5, 0.5], 3); // landmark 1: x missing
    f.set([0.5, 0.5, Number.NaN], 6); // landmark 2: only z missing
    const out = makeInput(f, { useZ: false });
    expect(out.length).toBe(N * 3);
    expect(Array.from(out.subarray(2, 4))).toEqual([0, 0.5]); // landmark 1 x, y
    expect(out[N * 2 + 0]).toBe(1);
    expect(out[N * 2 + 1]).toBe(0); // x missing
    expect(out[N * 2 + 2]).toBe(1); // z is ignored without useZ
    expect(Array.from(out).some(Number.isNaN)).toBe(false);
  });

  it("requires z to be finite when useZ is set", () => {
    const f = frame(0.5);
    f.set([0.5, 0.5, Number.NaN], 6);
    const out = makeInput(f, { useZ: true });
    expect(out[N * 3 + 2]).toBe(0);
    expect(Array.from(out.subarray(6, 9))).toEqual([0.5, 0.5, 0]);
  });

  it("turns an all-NaN frame into an all-zero (empty) row", () => {
    const out = makeInput(frame(Number.NaN), { useZ: false });
    expect(out.every((v) => v === 0)).toBe(true);
  });

  it("does not modify its input and rejects a partial frame", () => {
    const f = frame(0.25);
    const before = f.slice();
    makeInput(f, { useZ: false });
    expect(f).toEqual(before);
    expect(() => makeInput(new Float32Array(N * 3 + 1), { useZ: false })).toThrow(RangeError);
  });
});

describe("resampleNearest", () => {
  const ramp = (frames: number): Float32Array => Float32Array.from({ length: frames }, (_, i) => i);

  it("downsamples by taking the nearest earlier frame", () => {
    expect(Array.from(resampleNearest(ramp(8), 1, 4))).toEqual([0, 2, 4, 6]);
  });

  it("upsamples by repeating frames and never interpolates", () => {
    expect(Array.from(resampleNearest(ramp(3), 1, 6))).toEqual([0, 0, 1, 1, 2, 2]);
    expect(Array.from(resampleNearest(ramp(1), 1, 5))).toEqual([0, 0, 0, 0, 0]);
  });

  it("keeps NaN as NaN", () => {
    const a = Float32Array.from([1, Number.NaN, 3]);
    expect(Number.isNaN(resampleNearest(a, 1, 3)[1])).toBe(true);
  });

  it("rejects an empty sequence and a ragged length", () => {
    expect(() => resampleNearest(new Float32Array(0), 2, 4)).toThrow(RangeError);
    expect(() => resampleNearest(new Float32Array(5), 2, 4)).toThrow(RangeError);
  });
});
