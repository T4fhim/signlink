import { describe, expect, it } from "vitest";
import { RunStats } from "./stats.ts";

/** `count` frames `stepMs` apart, both hands in view unless `hands` says otherwise. */
function feed(
  stats: RunStats,
  count: number,
  stepMs: number,
  hands: { left: boolean; right: boolean } = { left: true, right: true },
  start = 1000,
): void {
  for (let i = 0; i < count; i++) {
    stats.frame({
      doneAt: start + i * stepMs,
      landmarkMs: 30,
      leftHand: hands.left,
      rightHand: hands.right,
    });
  }
}

describe("RunStats frames", () => {
  it("reports the whole-run mean fps, not a rolling window", () => {
    const stats = new RunStats();
    feed(stats, 2501, 40); // 100 s at 25 fps
    const report = stats.report();
    expect(report.frames).toBe(2501);
    expect(report.durationMs).toBeCloseTo(100_000, 6);
    expect(report.fps).toBeCloseTo(25, 6);
    expect(report.targets.fps).toEqual({ target: 25, minDurationMs: 60_000, met: true });
  });

  it("does not meet the fps target below 25 fps", () => {
    const stats = new RunStats();
    feed(stats, 1500, 50); // 20 fps for 75 s
    expect(stats.report().targets.fps.met).toBe(false);
  });

  it("does not meet the fps target on a run shorter than 60 s", () => {
    const stats = new RunStats();
    feed(stats, 900, 33); // ~30 fps for 30 s
    const report = stats.report();
    expect(report.fps).toBeGreaterThan(25);
    expect(report.targets.fps.met).toBe(false);
  });

  it("finds the longest stall between frames", () => {
    const stats = new RunStats();
    feed(stats, 10, 40);
    feed(stats, 10, 40, undefined, 1000 + 9 * 40 + 350);
    expect(stats.report().longestGapMs).toBeCloseTo(350, 6);
  });

  it("counts the share of frames with each hand in view", () => {
    const stats = new RunStats();
    feed(stats, 4, 40, { left: true, right: true });
    feed(stats, 4, 40, { left: false, right: true }, 5000);
    feed(stats, 2, 40, { left: false, right: false }, 6000);
    expect(stats.report().hands).toEqual({ left: 0.4, right: 0.8, either: 0.8, both: 0.4 });
  });

  it("summarizes landmark time per frame", () => {
    const stats = new RunStats();
    for (const ms of [10, 20, 30, 40, 100]) {
      stats.frame({ doneAt: ms, landmarkMs: ms, leftHand: true, rightHand: true });
    }
    expect(stats.report().landmarkMs).toEqual({ p50: 30, p95: 100, max: 100, count: 5 });
  });
});

describe("RunStats signs", () => {
  const sign = (tEnd: number, paintedAt: number | null, inferMs = 20) => ({
    tEnd,
    tDetected: tEnd + 220,
    receivedAt: tEnd + 300,
    inferMs,
    paintedAt,
    reason: "quiet" as const,
  });

  it("measures sign end to painted caption, for captioned signs only", () => {
    const stats = new RunStats();
    stats.sign(sign(1000, 1400));
    stats.sign(sign(2000, 2500));
    stats.sign(sign(3000, null)); // nothing captioned: no latency to a caption
    const report = stats.report();
    expect(report.signs).toEqual({ total: 3, captioned: 2, forced: 0 });
    expect(report.signEndToCaptionMs).toEqual({ p50: 400, p95: 500, max: 500, count: 2 });
    expect(report.targets.p95LatencyMs).toEqual({ target: 500, minSamples: 20, met: false });
  });

  it("meets the latency target when p95 is under 500 ms over enough captions", () => {
    const stats = new RunStats();
    for (let i = 0; i < 20; i++) stats.sign(sign(i * 1000, i * 1000 + 450));
    expect(stats.report().targets.p95LatencyMs.met).toBe(true);
  });

  it("does not call the target met on fewer than 20 captions, where p95 is only the maximum", () => {
    const stats = new RunStats();
    for (let i = 0; i < 19; i++) stats.sign(sign(i * 1000, i * 1000 + 450));
    const report = stats.report();
    expect(report.signEndToCaptionMs.p95).toBe(450);
    expect(report.targets.p95LatencyMs.met).toBe(false);
  });

  it("leaves signs cut at maxSignMs out of the latency numbers but still counts them", () => {
    const stats = new RunStats();
    for (let i = 0; i < 20; i++) stats.sign(sign(i * 1000, i * 1000 + 450));
    // A cut sign has no real end: its tEnd is about the moment it was cut, so its wait looks ~0.
    stats.sign({ ...sign(30_000, 30_050), reason: "max" });
    const report = stats.report();
    expect(report.signs).toEqual({ total: 21, captioned: 21, forced: 1 });
    expect(report.signEndToCaptionMs.count).toBe(20);
    expect(report.signEndToCaptionMs.p50).toBe(450);
    expect(report.decoderWaitMs.count).toBe(20);
  });

  it("splits the latency into decoder wait, pipeline and model time", () => {
    const stats = new RunStats();
    stats.sign(sign(1000, 1400, 35));
    const report = stats.report();
    expect(report.decoderWaitMs.p50).toBe(220); // sign end to the closing frame
    expect(report.pipelineMs.p50).toBe(80); // closing frame to the message arriving
    expect(report.inferMs.p50).toBe(35);
  });

  it("has no latency target met without a captioned sign", () => {
    const report = new RunStats().report();
    expect(report.signEndToCaptionMs.count).toBe(0);
    expect(report.signEndToCaptionMs.p95).toBeNaN();
    expect(report.targets.p95LatencyMs.met).toBe(false);
    expect(report.targets.fps.met).toBe(false);
    expect(report.fps).toBeNaN();
  });

  it("starts over on reset", () => {
    const stats = new RunStats();
    feed(stats, 10, 40);
    stats.sign(sign(1000, 1400));
    stats.reset();
    const report = stats.report();
    expect(report.frames).toBe(0);
    expect(report.signs.total).toBe(0);
  });

  it("produces a report that survives JSON (NaN becomes null, nothing throws)", () => {
    const stats = new RunStats();
    feed(stats, 5, 40);
    expect(() => JSON.stringify(stats.report())).not.toThrow();
  });
});
