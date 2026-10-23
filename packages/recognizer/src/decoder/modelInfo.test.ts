import { describe, expect, it } from "vitest";
import { parseModelInfo } from "./modelInfo.ts";

const valid = (): Record<string, unknown> => ({
  format: "slk-recognizer-v1",
  classes: ["hello", "thankyou", "<other>"],
  other_index: 2,
  input: {
    name: "windows",
    length: 64,
    dim: 438,
    use_z: false,
    velocity: false,
    n_landmarks: 146,
    layout: "slk-landmarks-v1",
  },
});
const withInput = (patch: Record<string, unknown>): Record<string, unknown> => ({
  ...valid(),
  input: { ...(valid()["input"] as object), ...patch },
});

describe("parseModelInfo", () => {
  it("reads what the decoder needs from model.json", () => {
    expect(parseModelInfo(valid())).toEqual({
      classes: ["hello", "thankyou", "<other>"],
      otherIndex: 2,
      length: 64,
      dim: 438,
      useZ: false,
      nLandmarks: 146,
    });
  });

  it("accepts the z variant when dim matches", () => {
    const info = parseModelInfo(withInput({ use_z: true, dim: 146 * 4 }));
    expect(info.useZ).toBe(true);
    expect(info.dim).toBe(584);
  });

  it.each([
    ["a non-object", 42],
    ["another format", { ...valid(), format: "slk-recognizer-v2" }],
    ["a velocity model (not ported to TS)", withInput({ velocity: true, dim: 438 * 2 })],
    ["another layout", withInput({ layout: "slk-landmarks-v2" })],
    ["another input name", withInput({ name: "x" })],
    ["a dim that does not match N and use_z", withInput({ dim: 400 })],
    ["a zero window length", withInput({ length: 0 })],
    ["an other_index that is not <other>", { ...valid(), other_index: 0 }],
    ["an other_index out of range", { ...valid(), other_index: 3 }],
    ["duplicate classes", { ...valid(), classes: ["a", "a", "<other>"] }],
    ["no classes", { ...valid(), classes: [] }],
    ["non-string classes", { ...valid(), classes: ["a", 1, "<other>"] }],
  ])("rejects %s", (_label, json) => {
    expect(() => parseModelInfo(json)).toThrow(/model\.json/);
  });
});
