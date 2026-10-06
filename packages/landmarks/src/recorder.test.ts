import { describe, expect, it } from "vitest";
import type { Presence } from "./frame.ts";
import type { FrameResult } from "./protocol.ts";
import { Recorder } from "./recorder.ts";

const LENGTH = 6; // a toy frame length

const present = (overrides: Partial<Presence> = {}): Presence => ({
  leftHand: false,
  rightHand: false,
  pose: true,
  face: true,
  ...overrides,
});

const frame = (t: number, fill: number, presence = present()): FrameResult => ({
  t,
  frame: new Float32Array(LENGTH).fill(fill),
  present: presence,
  timingsMs: { hand: 0, pose: 0, face: 0, total: 0 },
});

describe("Recorder", () => {
  it("is not complete until the target duration has elapsed", () => {
    const recorder = new Recorder(LENGTH, 3000);
    expect(recorder.add(frame(1000, 1))).toBe(false);
    expect(recorder.add(frame(3999, 2))).toBe(false);
    expect(recorder.isComplete).toBe(false);
    expect(recorder.add(frame(4000, 3))).toBe(true);
    expect(recorder.isComplete).toBe(true);
  });

  it("ignores frames after it completes", () => {
    const recorder = new Recorder(LENGTH, 100);
    recorder.add(frame(0, 1));
    recorder.add(frame(100, 2));
    expect(recorder.add(frame(200, 3))).toBe(true);
    expect(recorder.count).toBe(2);
  });

  it("concatenates frames row-major in arrival order", () => {
    const recorder = new Recorder(LENGTH, 1000);
    recorder.add(frame(0, 1));
    recorder.add(frame(40, 2));
    recorder.add(frame(80, 3));
    const all = recorder.toFloat32();
    expect(all.length).toBe(3 * LENGTH);
    expect(Array.from(all.filter((_, i) => i % LENGTH === 0))).toEqual([1, 2, 3]);
  });

  it("summarizes frames, duration, fps and how often each part was detected", () => {
    const recorder = new Recorder(LENGTH, 1000);
    recorder.add(frame(0, 0, present({ leftHand: true })));
    recorder.add(frame(40, 0, present({ leftHand: true, rightHand: true })));
    recorder.add(frame(80, 0, present({ pose: false })));
    recorder.add(frame(120, 0, present()));
    const s = recorder.summary();
    expect(s.frames).toBe(4);
    expect(s.durationMs).toBe(120);
    expect(s.fps).toBeCloseTo(25);
    expect(s.presence).toEqual({ leftHand: 0.5, rightHand: 0.25, pose: 0.75, face: 1 });
  });

  it("summarizes an empty recording without dividing by zero", () => {
    const s = new Recorder(LENGTH, 1000).summary();
    expect(s.frames).toBe(0);
    expect(s.fps).toBeNaN();
    expect(s.presence.face).toBe(0);
  });

  it("rejects a frame of the wrong length", () => {
    const recorder = new Recorder(LENGTH, 1000);
    const bad = { ...frame(0, 1), frame: new Float32Array(LENGTH - 1) };
    expect(() => recorder.add(bad)).toThrow(RangeError);
  });
});
