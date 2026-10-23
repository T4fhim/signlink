import { LANDMARK_COUNT } from "@signlnk/landmarks";
import { describe, expect, it } from "vitest";
import {
  Endpointer,
  handIndices,
  handSpeed,
  type EndpointerConfig,
  type Segment,
} from "./endpointer.ts";

const N = LANDMARK_COUNT;
const FRAME = N * 3;
const HANDS = handIndices();

// Deterministic: no smoothing, round numbers. Speeds are in shoulder widths per second.
const BASE: EndpointerConfig = {
  startSpeed: 0.5,
  stopSpeed: 0.25,
  startMs: 60,
  stopMs: 200,
  prerollMs: 100,
  tailMs: 100,
  minSignMs: 200,
  maxSignMs: 3000,
  smoothing: 1,
};

/** A normalized frame: both hands at `x` (spread a little), or no hands. */
function frame(handsAt: number | null): Float32Array {
  const f = new Float32Array(FRAME).fill(Number.NaN);
  if (handsAt !== null) {
    for (const [k, i] of HANDS.entries()) {
      f[i * 3] = handsAt + 0.01 * k;
      f[i * 3 + 1] = 0.2;
      f[i * 3 + 2] = 0;
    }
  }
  return f;
}

type Hands = "none" | "still" | "move" | "nan";
interface Phase {
  ms: number;
  hands: Hands;
}

function run(
  phases: Phase[],
  options: { fps?: number; speed?: number; config?: Partial<EndpointerConfig> } = {},
): Segment[] {
  const { fps = 30, speed = 1.5, config = {} } = options;
  const endpointer = new Endpointer({ ...BASE, ...config });
  const dt = 1000 / fps;
  const segments: Segment[] = [];
  let t = 1000;
  let x = 0.1;
  for (const phase of phases) {
    const end = t + phase.ms;
    for (; t < end; t += dt) {
      if (phase.hands === "move") x += (speed * dt) / 1000;
      const f =
        phase.hands === "nan"
          ? new Float32Array(FRAME).fill(Number.NaN)
          : frame(phase.hands === "none" ? null : x);
      const segment = endpointer.push(t, f);
      if (segment) segments.push(segment);
    }
  }
  return segments;
}

describe("handIndices", () => {
  it("is both hands, the first 42 landmarks of slk-landmarks-v1", () => {
    expect(handIndices()).toEqual(Array.from({ length: 42 }, (_, i) => i));
  });
});

describe("handSpeed", () => {
  const at = (x: number, only?: number[]): Float32Array => {
    const f = new Float32Array(FRAME).fill(Number.NaN);
    for (const i of only ?? HANDS) f.set([x, 0.2, 0], i * 3);
    return f;
  };

  it("is the mean hand displacement per second", () => {
    expect(handSpeed(at(0.1), at(0.2), 100, HANDS)).toBeCloseTo(1.0, 6);
  });

  it("uses only landmarks present in both frames", () => {
    expect(handSpeed(at(0.1, [0, 1]), at(0.2, [1, 2]), 100, HANDS)).toBeCloseTo(1.0, 6);
  });

  it("is 0 when no hand landmark is shared, or time did not advance", () => {
    expect(handSpeed(at(0.1, [0]), at(0.9, [5]), 100, HANDS)).toBe(0);
    expect(handSpeed(at(0.1), frame(null), 100, HANDS)).toBe(0);
    expect(handSpeed(at(0.1), at(0.2), 0, HANDS)).toBe(0);
  });
});

describe("Endpointer", () => {
  it("emits nothing while the hands are absent or still", () => {
    expect(run([{ ms: 3000, hands: "none" }])).toEqual([]);
    expect(run([{ ms: 3000, hands: "still" }])).toEqual([]);
  });

  it("emits one segment per sign, ending stopMs after the last active frame", () => {
    const dt = 1000 / 30;
    const segments = run([
      { ms: 500, hands: "none" },
      { ms: 800, hands: "move" },
      { ms: 600, hands: "still" },
    ]);
    expect(segments).toHaveLength(1);
    const s = segments[0] as Segment;
    expect(s.reason).toBe("quiet");
    // a hand needs two frames in view before it has a speed, so the start is up to 2 frames late
    expect(s.tStart).toBeGreaterThanOrEqual(1500 - 1);
    expect(s.tStart).toBeLessThanOrEqual(1500 + 2 * dt + 1);
    expect(s.tEnd).toBeGreaterThan(2300 - dt - 1);
    expect(s.tEnd).toBeLessThan(2300 + 1);
    expect(s.tDetected - s.tEnd).toBeGreaterThanOrEqual(200);
    expect(s.tDetected - s.tEnd).toBeLessThanOrEqual(200 + dt + 1);
  });

  it("keeps preroll and tail frames around the sign", () => {
    const s = run([
      { ms: 500, hands: "still" },
      { ms: 800, hands: "move" },
      { ms: 600, hands: "still" },
    ])[0] as Segment;
    expect(s.frames.length).toBe(s.frameCount * FRAME);
    // about prerollMs + the sign + tailMs = 1000 ms of 30 fps frames
    expect(s.frameCount).toBeGreaterThanOrEqual(28);
    expect(s.frameCount).toBeLessThanOrEqual(33);
  });

  it("drops a movement shorter than minSignMs", () => {
    expect(
      run([
        { ms: 300, hands: "still" },
        { ms: 100, hands: "move" },
        { ms: 600, hands: "still" },
      ]),
    ).toEqual([]);
  });

  it("does not split a sign at a dropout shorter than stopMs", () => {
    const segments = run([
      { ms: 400, hands: "move" },
      { ms: 66, hands: "none" },
      { ms: 400, hands: "move" },
      { ms: 600, hands: "still" },
    ]);
    expect(segments).toHaveLength(1);
  });

  it("splits two signs separated by a pause longer than stopMs", () => {
    const [first, second, ...rest] = run([
      { ms: 500, hands: "move" },
      { ms: 500, hands: "still" },
      { ms: 500, hands: "move" },
      { ms: 500, hands: "still" },
    ]);
    expect(rest).toEqual([]);
    expect(first && second && second.tStart > first.tEnd).toBe(true);
  });

  it("treats frames with no usable shoulders (all NaN) as quiet", () => {
    const segments = run([
      { ms: 500, hands: "move" },
      { ms: 600, hands: "nan" },
    ]);
    expect(segments).toHaveLength(1);
  });

  it("force-closes a segment at maxSignMs and keeps listening", () => {
    const segments = run(
      [
        { ms: 2500, hands: "move" },
        { ms: 600, hands: "still" },
      ],
      { config: { maxSignMs: 1000 } },
    );
    expect(segments.length).toBeGreaterThanOrEqual(2);
    expect((segments[0] as Segment).reason).toBe("max");
    // closes on the first frame at or past maxSignMs, so at most one frame over
    expect((segments[0] as Segment).tEnd - (segments[0] as Segment).tStart).toBeLessThanOrEqual(
      1000 + 1000 / 30 + 1,
    );
    expect((segments.at(-1) as Segment).reason).toBe("quiet");
  });

  it("finds the same sign boundaries at 15 and 30 fps", () => {
    const script: Phase[] = [
      { ms: 500, hands: "still" },
      { ms: 800, hands: "move" },
      { ms: 600, hands: "still" },
    ];
    const [fast] = run(script, { fps: 30 }) as [Segment];
    const [slow] = run(script, { fps: 15 }) as [Segment];
    expect(Math.abs(fast.tStart - slow.tStart)).toBeLessThanOrEqual(70);
    expect(Math.abs(fast.tEnd - slow.tEnd)).toBeLessThanOrEqual(70);
  });

  it("reset() abandons an open segment", () => {
    const endpointer = new Endpointer(BASE);
    let t = 0;
    let x = 0.1;
    for (; t < 400; t += 33) {
      x += 0.05;
      endpointer.push(t, frame(x));
    }
    endpointer.reset();
    const out: Segment[] = [];
    for (; t < 1500; t += 33) {
      const s = endpointer.push(t, frame(x));
      if (s) out.push(s);
    }
    expect(out).toEqual([]);
  });
});
