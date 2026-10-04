import { describe, expect, it } from "vitest";
import { assembleFrame, type Point } from "./frame.ts";
import { FRAME_LENGTH, LAYOUT } from "./layout.ts";

/** Points whose x encodes `base + index`, so a value in the frame reveals where it came from. */
const points = (count: number, base: number): Point[] =>
  Array.from({ length: count }, (_, i) => ({ x: base + i, y: -(base + i), z: 0.5 }));

const LEFT_HAND = 1000;
const RIGHT_HAND = 2000;
const POSE = 3000;
const FACE = 4000;

/** Start offset of a named group in the flat landmark order. */
const offsetOf = (name: string): number => {
  let offset = 0;
  for (const group of LAYOUT.groups) {
    if (group.name === name) return offset;
    offset += group.indices.length;
  }
  throw new Error(`no group ${name}`);
};
const xAt = (frame: Float32Array, landmark: number): number => frame[landmark * 3] ?? Number.NaN;

const full = {
  hands: [
    { label: "Left" as const, points: points(21, LEFT_HAND) },
    { label: "Right" as const, points: points(21, RIGHT_HAND) },
  ],
  pose: points(33, POSE),
  face: points(478, FACE),
};

describe("assembleFrame", () => {
  it("produces a float32 frame of N * 3 values", () => {
    const { frame } = assembleFrame(full, LAYOUT, { swapHandedness: false });
    expect(frame).toBeInstanceOf(Float32Array);
    expect(frame.length).toBe(FRAME_LENGTH);
    expect(frame.every(Number.isFinite)).toBe(true);
  });

  it("treats MediaPipe's handedness labels as the signer's own side by default", () => {
    const { frame, present } = assembleFrame(full, LAYOUT, { swapHandedness: false });
    expect(xAt(frame, offsetOf("hand_left") + 5)).toBe(LEFT_HAND + 5);
    expect(xAt(frame, offsetOf("hand_right") + 5)).toBe(RIGHT_HAND + 5);
    expect(present).toEqual({ leftHand: true, rightHand: true, pose: true, face: true });
  });

  it("swaps the labels when asked", () => {
    const { frame } = assembleFrame(full, LAYOUT, { swapHandedness: true });
    expect(xAt(frame, offsetOf("hand_left") + 5)).toBe(RIGHT_HAND + 5);
    expect(xAt(frame, offsetOf("hand_right") + 5)).toBe(LEFT_HAND + 5);
  });

  it("copies the pose and face subsets by their source indices", () => {
    const { frame } = assembleFrame(full, LAYOUT, { swapHandedness: false });
    const pose = LAYOUT.groups.find((g) => g.name === "pose");
    pose?.indices.forEach((source, i) => {
      expect(xAt(frame, offsetOf("pose") + i)).toBe(POSE + source);
    });
    const lips = LAYOUT.groups.find((g) => g.name === "face_lips");
    lips?.indices.forEach((source, i) => {
      expect(xAt(frame, offsetOf("face_lips") + i)).toBe(FACE + source);
    });
  });

  it("writes y and z alongside x", () => {
    const { frame } = assembleFrame(full, LAYOUT, { swapHandedness: false });
    const at = offsetOf("hand_left") * 3;
    expect([frame[at], frame[at + 1], frame[at + 2]]).toEqual([LEFT_HAND, -LEFT_HAND, 0.5]);
  });

  it("fills undetected parts with NaN and reports presence", () => {
    const { frame, present } = assembleFrame({ hands: [], pose: null, face: full.face }, LAYOUT, {
      swapHandedness: false,
    });
    expect(present).toEqual({ leftHand: false, rightHand: false, pose: false, face: true });
    const handStart = offsetOf("hand_left") * 3;
    const poseStart = offsetOf("pose") * 3;
    expect(frame.slice(handStart, poseStart).every(Number.isNaN)).toBe(true);
    expect(Number.isFinite(xAt(frame, offsetOf("face_lips")))).toBe(true);
  });

  it("uses only the first hand when two share a side", () => {
    const twoLeft = [
      { label: "Left" as const, points: points(21, 100) },
      { label: "Left" as const, points: points(21, 200) },
    ];
    const { frame } = assembleFrame({ hands: twoLeft, pose: null, face: null }, LAYOUT, {
      swapHandedness: false,
    });
    expect(xAt(frame, offsetOf("hand_left"))).toBe(100);
  });

  it("turns non-finite or out-of-range points into NaN", () => {
    const bad = points(21, 0);
    bad[3] = { x: Number.NaN, y: 0, z: 0 };
    const short = points(5, 0); // fewer points than the layout asks for
    const hands = [
      { label: "Left" as const, points: bad },
      { label: "Right" as const, points: short },
    ];
    const { frame } = assembleFrame({ hands, pose: null, face: null }, LAYOUT, {
      swapHandedness: false,
    });
    expect(Number.isNaN(xAt(frame, offsetOf("hand_left") + 3))).toBe(true);
    expect(Number.isFinite(xAt(frame, offsetOf("hand_left") + 2))).toBe(true);
    expect(Number.isFinite(xAt(frame, offsetOf("hand_right") + 4))).toBe(true);
    expect(Number.isNaN(xAt(frame, offsetOf("hand_right") + 5))).toBe(true);
  });
});
