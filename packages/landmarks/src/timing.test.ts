import { describe, expect, it } from "vitest";
import { RollingWindow, fpsFromTimestamps } from "./timing.ts";

describe("RollingWindow", () => {
  it("reports NaN when empty", () => {
    expect(new RollingWindow(5).median).toBeNaN();
  });

  it("uses nearest-rank percentiles", () => {
    const w = new RollingWindow(100);
    for (let i = 1; i <= 100; i++) w.push(i);
    expect(w.median).toBe(50);
    expect(w.p95).toBe(95);
    expect(w.percentile(100)).toBe(100);
  });

  it("drops the oldest samples beyond capacity", () => {
    const w = new RollingWindow(3);
    [1, 2, 3, 4].forEach((v) => w.push(v));
    expect(w.count).toBe(3);
    expect(w.percentile(0)).toBe(2);
  });
});

describe("fpsFromTimestamps", () => {
  it("computes frames per second from frame completion times", () => {
    expect(fpsFromTimestamps([0, 40, 80, 120])).toBeCloseTo(25);
  });

  it("is NaN with too little data", () => {
    expect(fpsFromTimestamps([])).toBeNaN();
    expect(fpsFromTimestamps([10])).toBeNaN();
    expect(fpsFromTimestamps([10, 10])).toBeNaN();
  });
});
