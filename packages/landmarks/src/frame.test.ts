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
    const { frame } = assembleFrame(full, LAYOUT, {
      handAssignment: "label",
      swapHandedness: false,
    });
    expect(frame).toBeInstanceOf(Float32Array);
    expect(frame.length).toBe(FRAME_LENGTH);
    expect(frame.every(Number.isFinite)).toBe(true);
  });

  it("treats MediaPipe's handedness labels as the signer's own side by default", () => {
    const { frame, present } = assembleFrame(full, LAYOUT, {
      handAssignment: "label",
      swapHandedness: false,
    });
    expect(xAt(frame, offsetOf("hand_left") + 5)).toBe(LEFT_HAND + 5);
    expect(xAt(frame, offsetOf("hand_right") + 5)).toBe(RIGHT_HAND + 5);
    expect(present).toEqual({ leftHand: true, rightHand: true, pose: true, face: true });
  });

  it("swaps the labels when asked", () => {
    const { frame } = assembleFrame(full, LAYOUT, {
      handAssignment: "label",
      swapHandedness: true,
    });
    expect(xAt(frame, offsetOf("hand_left") + 5)).toBe(RIGHT_HAND + 5);
    expect(xAt(frame, offsetOf("hand_right") + 5)).toBe(LEFT_HAND + 5);
  });

  it("copies the pose and face subsets by their source indices", () => {
    const { frame } = assembleFrame(full, LAYOUT, {
      handAssignment: "label",
      swapHandedness: false,
    });
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
    const { frame } = assembleFrame(full, LAYOUT, {
      handAssignment: "label",
      swapHandedness: false,
    });
    const at = offsetOf("hand_left") * 3;
    expect([frame[at], frame[at + 1], frame[at + 2]]).toEqual([LEFT_HAND, -LEFT_HAND, 0.5]);
  });

  it("fills undetected parts with NaN and reports presence", () => {
    const { frame, present } = assembleFrame({ hands: [], pose: null, face: full.face }, LAYOUT, {
      handAssignment: "label",
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
      handAssignment: "label",
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
      handAssignment: "label",
      swapHandedness: false,
    });
    expect(Number.isNaN(xAt(frame, offsetOf("hand_left") + 3))).toBe(true);
    expect(Number.isFinite(xAt(frame, offsetOf("hand_left") + 2))).toBe(true);
    expect(Number.isFinite(xAt(frame, offsetOf("hand_right") + 4))).toBe(true);
    expect(Number.isNaN(xAt(frame, offsetOf("hand_right") + 5))).toBe(true);
  });
});

describe("assembleFrame hand assignment by pose wrist", () => {
  const nan = { x: Number.NaN, y: Number.NaN, z: Number.NaN };
  /** 33 pose points, NaN except the wrists (BlazePose 15 = left, 16 = right). */
  const poseWithWrists = (
    left: [number, number] | null,
    right: [number, number] | null,
  ): Point[] => {
    const pose: Point[] = Array.from({ length: 33 }, () => nan);
    if (left) pose[15] = { x: left[0], y: left[1], z: 0 };
    if (right) pose[16] = { x: right[0], y: right[1], z: 0 };
    return pose;
  };
  /** A hand whose wrist is at (x, y); other points encode `base + index` in x. */
  const handAt = (x: number, y: number, base: number): Point[] => {
    const hand = points(21, base);
    hand[0] = { x, y, z: 0 };
    return hand;
  };
  const LEFT_WRIST: [number, number] = [0.7, 0.8]; // the subject's left is on the image's right
  const RIGHT_WRIST: [number, number] = [0.3, 0.8];
  const wrist = { handAssignment: "wrist", swapHandedness: false } as const;
  const handSlots = (frame: Float32Array) => ({
    left: xAt(frame, offsetOf("hand_left") + 5),
    right: xAt(frame, offsetOf("hand_right") + 5),
  });

  it("gives a lone hand to the nearer wrist even when MediaPipe labelled it the other side", () => {
    const hand = { label: "Left" as const, points: handAt(0.31, 0.79, 100) }; // on the right wrist
    const raw = { hands: [hand], pose: poseWithWrists(LEFT_WRIST, RIGHT_WRIST), face: null };
    const { frame, present } = assembleFrame(raw, LAYOUT, wrist);
    expect(handSlots(frame).right).toBe(105);
    expect(Number.isNaN(handSlots(frame).left)).toBe(true);
    expect(present).toMatchObject({ leftHand: false, rightHand: true });
  });

  it("gives a lone hand near the left wrist to the left slot despite a Right label", () => {
    const hand = { label: "Right" as const, points: handAt(0.72, 0.8, 100) };
    const raw = { hands: [hand], pose: poseWithWrists(LEFT_WRIST, RIGHT_WRIST), face: null };
    const slots = handSlots(assembleFrame(raw, LAYOUT, wrist).frame);
    expect(slots.left).toBe(105);
    expect(Number.isNaN(slots.right)).toBe(true);
  });

  it("corrects two hands whose labels are both swapped", () => {
    const hands = [
      { label: "Right" as const, points: handAt(0.7, 0.8, 100) }, // really the left hand
      { label: "Left" as const, points: handAt(0.3, 0.8, 200) }, // really the right hand
    ];
    const raw = { hands, pose: poseWithWrists(LEFT_WRIST, RIGHT_WRIST), face: null };
    expect(handSlots(assembleFrame(raw, LAYOUT, wrist).frame)).toEqual({ left: 105, right: 205 });
  });

  it("keeps two correctly labelled hands where they are", () => {
    const hands = [
      { label: "Left" as const, points: handAt(0.7, 0.8, 100) },
      { label: "Right" as const, points: handAt(0.3, 0.8, 200) },
    ];
    const raw = { hands, pose: poseWithWrists(LEFT_WRIST, RIGHT_WRIST), face: null };
    expect(handSlots(assembleFrame(raw, LAYOUT, wrist).frame)).toEqual({ left: 105, right: 205 });
  });

  it("falls back to the label when a pose wrist is missing or there is no pose", () => {
    const hand = { label: "Left" as const, points: handAt(0.31, 0.79, 100) };
    for (const pose of [poseWithWrists(LEFT_WRIST, null), poseWithWrists(null, null), null]) {
      const frame = assembleFrame({ hands: [hand], pose, face: null }, LAYOUT, wrist).frame;
      expect(handSlots(frame).left).toBe(105);
    }
  });

  it("falls back to the label on an exact tie", () => {
    // 0.25, 0.5 and 0.75 are exact in binary, so the two distances are exactly equal
    const hand = { label: "Right" as const, points: handAt(0.5, 0.8, 100) }; // midway
    const pose = poseWithWrists([0.75, 0.8], [0.25, 0.8]);
    const raw = { hands: [hand], pose, face: null };
    expect(handSlots(assembleFrame(raw, LAYOUT, wrist).frame).right).toBe(105);
  });

  it('ignores the wrists when asked to use labels ("label")', () => {
    const hand = { label: "Left" as const, points: handAt(0.31, 0.79, 100) };
    const raw = { hands: [hand], pose: poseWithWrists(LEFT_WRIST, RIGHT_WRIST), face: null };
    const options = { handAssignment: "label", swapHandedness: false } as const;
    expect(handSlots(assembleFrame(raw, LAYOUT, options).frame).left).toBe(105);
  });
});
