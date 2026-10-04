import { describe, expect, it } from "vitest";
import { LANDMARK_LAYOUT_ID } from "./index";

describe("landmarks package", () => {
  it("exposes the layout id", () => {
    expect(LANDMARK_LAYOUT_ID).toBe("slk-landmarks-v1");
  });
});
