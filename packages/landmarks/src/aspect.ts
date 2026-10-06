import type { LandmarkLayoutV1 } from "@signlnk/schemas/landmark_layout_v1";
import { LAYOUT } from "./layout.ts";

/**
 * Converts landmarks to the layout's reference aspect ratio (ADR-0007).
 *
 * MediaPipe normalizes x by image width and y by image height, so the same face has different
 * vertical proportions on a 4:3 webcam and on the portrait phone video Kaggle ISLR was recorded on.
 * Mirrors ml/signlnk_ml/features/aspect.py; tests/fixtures/aspect/cases.json is the parity contract.
 */

/** Multiplier for y that moves a `width x height` frame to the layout's reference aspect ratio. */
export function aspectScale(
  width: number,
  height: number,
  layout: LandmarkLayoutV1 = LAYOUT,
): number {
  if (!(width > 0) || !(height > 0)) throw new RangeError(`invalid image size ${width}x${height}`);
  const reference = layout.reference_aspect;
  if (reference === undefined)
    throw new Error(`layout ${layout.layout_id} has no reference_aspect`);
  return reference / (width / height);
}

/**
 * Rescales y of a frame (float32 [N * 3]) or clip (float32 [T * N * 3]) in place and returns it.
 * x and z are unchanged and NaN stays NaN. Math runs in float64, rounded to float32 on store.
 */
export function toReferenceAspect(
  landmarks: Float32Array,
  width: number,
  height: number,
  layout: LandmarkLayoutV1 = LAYOUT,
): Float32Array {
  const scale = aspectScale(width, height, layout);
  for (let i = 1; i < landmarks.length; i += 3) {
    landmarks[i] = (landmarks[i] ?? Number.NaN) * scale;
  }
  return landmarks;
}
