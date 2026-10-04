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
   * Swap MediaPipe's Left/Right labels. Default false: measured on the benchmark laptop
   * (2026-10-04, /dev/landmarks), the labels MediaPipe Tasks returns for raw, unmirrored camera
   * frames already name the signer's own hand (raising the right hand reported "right").
   * [VERIFY] for any horizontally mirrored input, which this pipeline does not use.
   */
  swapHandedness: boolean;
}

const signerSide = (label: "Left" | "Right", swap: boolean): "Left" | "Right" => {
  if (!swap) return label;
  return label === "Left" ? "Right" : "Left";
};

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
  for (const hand of raw.hands) {
    if (signerSide(hand.label, options.swapHandedness) === "Left") left ??= hand.points;
    else right ??= hand.points;
  }

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
