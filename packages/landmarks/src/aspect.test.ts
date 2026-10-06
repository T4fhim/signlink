import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { aspectScale, toReferenceAspect } from "./aspect.ts";
import { LAYOUT } from "./layout.ts";

const TOLERANCE = 1e-5; // same parity budget as normalization

interface Case {
  name: string;
  width: number;
  height: number;
  shape: [number, number, number];
  input: (number | null)[];
  expected: (number | null)[];
}
const cases = JSON.parse(
  readFileSync(new URL("../../../tests/fixtures/aspect/cases.json", import.meta.url), "utf8"),
) as Case[];
const toFloat32 = (values: (number | null)[]): Float32Array =>
  Float32Array.from(values, (v) => v ?? Number.NaN);

describe("aspectScale", () => {
  it("uses the layout's reference aspect", () => {
    expect(LAYOUT.reference_aspect).toBe(0.78);
  });

  it("is reference_aspect / (width / height)", () => {
    expect(aspectScale(640, 480)).toBeCloseTo(0.585, 12); // 4:3 webcam
    expect(aspectScale(1000, 1000)).toBeCloseTo(0.78, 12); // square
    expect(aspectScale(480, 640)).toBeCloseTo(0.78 / 0.75, 12); // 3:4 portrait
  });

  it("is exactly 1 when the image already has the reference aspect", () => {
    expect(aspectScale(780, 1000)).toBeCloseTo(1, 12);
  });

  it("rejects an invalid image size", () => {
    expect(() => aspectScale(0, 480)).toThrow(RangeError);
    expect(() => aspectScale(640, -1)).toThrow(RangeError);
    expect(() => aspectScale(Number.NaN, 480)).toThrow(RangeError);
  });

  it("needs a layout that has a reference aspect", () => {
    const without = { ...LAYOUT };
    delete without.reference_aspect;
    expect(() => aspectScale(640, 480, without)).toThrow("reference_aspect");
  });
});

describe("toReferenceAspect", () => {
  it("rescales y in place and leaves x, z and NaN alone", () => {
    const nan = Number.NaN;
    const frame = Float32Array.of(0.5, 0.4, 0.1, nan, 0.8, nan, 0.2, nan, 0.3);
    const out = toReferenceAspect(frame, 640, 480);
    expect(out).toBe(frame); // in place
    expect(out[0]).toBeCloseTo(0.5, 6);
    expect(out[1]).toBeCloseTo(0.4 * 0.585, 6);
    expect(out[2]).toBeCloseTo(0.1, 6);
    expect(Number.isNaN(out[3])).toBe(true);
    expect(out[4]).toBeCloseTo(0.8 * 0.585, 6);
    expect(Number.isNaN(out[5])).toBe(true);
    expect(out[6]).toBeCloseTo(0.2, 6);
    expect(Number.isNaN(out[7])).toBe(true);
    expect(out[8]).toBeCloseTo(0.3, 6);
  });
});

describe("aspect conversion parity with the Python reference", () => {
  it("has the four golden cases", () => {
    expect(cases.map((c) => c.name)).toEqual(["640x480", "1280x720", "480x640", "780x1000"]);
  });

  it.each(cases.map((c) => [c.name, c] as const))("matches golden case %s", (_name, c) => {
    const actual = toReferenceAspect(toFloat32(c.input), c.width, c.height);
    const expected = toFloat32(c.expected);
    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < expected.length; i++) {
      const a = actual[i] ?? Number.NaN;
      const e = expected[i] ?? Number.NaN;
      if (Number.isNaN(e)) expect(Number.isNaN(a), `[${i}] should be NaN`).toBe(true);
      else expect(Math.abs(a - e), `[${i}]: ${a} vs ${e}`).toBeLessThanOrEqual(TOLERANCE);
    }
  });
});
