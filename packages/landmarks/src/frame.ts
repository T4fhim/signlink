import type { LandmarkLayoutV1 } from "@signlnk/schemas/landmark_layout_v1";

export interface Point {
  x: number;
  y: number;
  z: number;
}

/** MediaPipe handedness label for one detected hand. */
export interface DetectedHand {
  label: "Left" | "Right";
  points: readonly Point[];
}

/** Raw per-frame detections from the three MediaPipe landmarkers. */
export interface RawDetections {
  hands: readonly DetectedHand[];
  /** 33 BlazePose landmarks, or null when no person was found. */
  pose: readonly Point[] | null;
  /** Face Landmarker points (478 incl. iris), or null when no face was found. */
  face: readonly Point[] | null;
}

export interface Presence {
  leftHand: boolean;
  rightHand: boolean;
  pose: boolean;
  face: boolean;
}

export interface AssembleOptions {
  /**
   * How a detected hand is given to the left or right slot.
   * - "wrist": to the nearer BlazePose wrist, as Holistic does, which produced the training data.
   *   MediaPipe Tasks' own handedness label was wrong in about 7% of single-hand frames on real
   *   recordings (the hand sat on the other wrist; ADR-0008). Falls back to the label when a pose
   *   wrist is missing, or on an exact tie.
   * - "label": MediaPipe's handedness label.
   */
  handAssignment: "wrist" | "label";
  /**
   * Swap MediaPipe's Left/Right labels when labels are used. Default false: measured on the
   * benchmark laptop (2026-10-04, /dev/landmarks), the labels MediaPipe Tasks returns for raw,
   * unmirrored camera frames name the signer's own hand. [VERIFY] for any mirrored input, which this
   * pipeline does not use.
   */
  swapHandedness: boolean;
}

type Side = "Left" | "Right";

const POSE_LEFT_WRIST = 15;
const POSE_RIGHT_WRIST = 16;

const signerSide = (label: Side, swap: boolean): Side => {
  if (!swap) return label;
  return label === "Left" ? "Right" : "Left";
};

const usable = (points: readonly Point[] | null, index: number): Point | null => {
  const p = points?.[index];
  return p && Number.isFinite(p.x) && Number.isFinite(p.y) ? p : null;
};

const distance = (a: Point, b: Point): number => Math.hypot(a.x - b.x, a.y - b.y);

/** The side of each hand, in the order of `hands`. */
function assignSides(
  hands: readonly DetectedHand[],
  pose: readonly Point[] | null,
  options: AssembleOptions,
): Side[] {
  const byLabel = hands.map((hand) => signerSide(hand.label, options.swapHandedness));
  if (options.handAssignment === "label") return byLabel;

  const leftWrist = usable(pose, POSE_LEFT_WRIST);
  const rightWrist = usable(pose, POSE_RIGHT_WRIST);
  const roots = hands.map((hand) => usable(hand.points, 0));
  if (!leftWrist || !rightWrist || roots.some((root) => root === null)) return byLabel;

  if (roots.length === 1 && roots[0]) {
    const toLeft = distance(roots[0], leftWrist);
    const toRight = distance(roots[0], rightWrist);
    return toLeft === toRight ? byLabel : [toLeft < toRight ? "Left" : "Right"];
  }
  const [first, second] = roots;
  if (roots.length === 2 && first && second) {
    const straight = distance(first, leftWrist) + distance(second, rightWrist);
    const crossed = distance(first, rightWrist) + distance(second, leftWrist);
    if (straight === crossed) return byLabel;
    return straight < crossed ? ["Left", "Right"] : ["Right", "Left"];
  }
  return byLabel;
}

/**
 * Packs raw detections into one flat float32 frame [N * 3] following the layout. Missing landmarks
 * (undetected part, index out of range, non-finite value) are NaN.
 */
export function assembleFrame(
  raw: RawDetections,
  layout: LandmarkLayoutV1,
  options: AssembleOptions,
): { frame: Float32Array; present: Presence } {
  let left: readonly Point[] | null = null;
  let right: readonly Point[] | null = null;
  const sides = assignSides(raw.hands, raw.pose, options);
  raw.hands.forEach((hand, i) => {
    if (sides[i] === "Left") left ??= hand.points;
    else right ??= hand.points;
  });

  const frame = new Float32Array(layout.n_landmarks * 3).fill(Number.NaN);
  let cursor = 0;
  for (const group of layout.groups) {
    let points: readonly Point[] | null;
    switch (group.source) {
      case "hand_left":
        points = left;
        break;
      case "hand_right":
        points = right;
        break;
      case "pose":
        points = raw.pose;
        break;
      case "face":
        points = raw.face;
        break;
    }
    for (const index of group.indices) {
      const p = points?.[index];
      if (p && Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z)) {
        frame[cursor] = p.x;
        frame[cursor + 1] = p.y;
        frame[cursor + 2] = p.z;
      }
      cursor += 3;
    }
  }

  return {
    frame,
    present: {
      leftHand: left !== null,
      rightHand: right !== null,
      pose: raw.pose !== null,
      face: raw.face !== null,
    },
  };
}
