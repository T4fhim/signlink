import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { LANDMARK_COUNT } from "./layout.ts";
import { decodeNpy, encodeNpy } from "./npy.ts";
import { PATTERN_FRAMES, patternArray } from "./npyPattern.ts";

const headerOf = (bytes: Uint8Array): string => {
  const length = new DataView(bytes.buffer, bytes.byteOffset).getUint16(8, true);
  return new TextDecoder().decode(bytes.subarray(10, 10 + length));
};

describe("encodeNpy", () => {
  it("writes the NumPy 1.0 header with a 64-byte aligned data start", () => {
    const bytes = encodeNpy(new Float32Array(6), [2, 3]);
    expect(Array.from(bytes.subarray(0, 8))).toEqual([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
    const header = headerOf(bytes);
    expect(header.startsWith("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }")).toBe(
      true,
    );
    expect(header.endsWith("\n")).toBe(true);
    expect((10 + header.length) % 64).toBe(0);
    expect(bytes.length).toBe(10 + header.length + 6 * 4);
  });

  it("writes a one-dimensional shape with a trailing comma", () => {
    expect(headerOf(encodeNpy(new Float32Array(3), [3]))).toContain("'shape': (3,)");
  });

  it("stores values little-endian and keeps NaN", () => {
    const bytes = encodeNpy(Float32Array.of(1.5, Number.NaN), [2]);
    const view = new DataView(bytes.buffer);
    const start = bytes.length - 8;
    expect(view.getFloat32(start, true)).toBe(1.5);
    expect(Number.isNaN(view.getFloat32(start + 4, true))).toBe(true);
    expect(Array.from(bytes.subarray(start, start + 4))).toEqual([0x00, 0x00, 0xc0, 0x3f]);
  });

  it("round-trips through decodeNpy", () => {
    const data = Float32Array.of(0, -1.25, Number.NaN, 3e7, 1e-7, 42);
    const decoded = decodeNpy(encodeNpy(data, [2, 3]));
    expect(decoded.shape).toEqual([2, 3]);
    expect(Array.from(decoded.data).map(String)).toEqual(Array.from(data).map(String));
  });

  it("rejects a shape that does not match the data", () => {
    expect(() => encodeNpy(new Float32Array(5), [2, 3])).toThrow(RangeError);
  });
});

describe("decodeNpy", () => {
  it("rejects bytes that are not an .npy file", () => {
    expect(() => decodeNpy(new Uint8Array(32))).toThrow("not an .npy file");
  });
});

describe("golden pattern shared with the Python loader", () => {
  const goldenUrl = new URL("../../../tests/fixtures/npy/pattern.npy", import.meta.url);
  const shape = [PATTERN_FRAMES, LANDMARK_COUNT, 3];

  it("is byte-identical to what encodeNpy produces (run `write-npy-fixture` if this fails)", () => {
    const expected = encodeNpy(patternArray(LANDMARK_COUNT), shape);
    const golden = readFileSync(goldenUrl);
    expect(golden.length).toBe(expected.length);
    expect(golden.equals(Buffer.from(expected))).toBe(true);
  });

  it("decodes to the pattern", () => {
    const decoded = decodeNpy(new Uint8Array(readFileSync(goldenUrl)));
    const expected = patternArray(LANDMARK_COUNT);
    expect(decoded.shape).toEqual(shape);
    expect(decoded.data.length).toBe(expected.length);
    expect(decoded.data.every((value, i) => Object.is(value, expected[i]))).toBe(true);
  });
});
