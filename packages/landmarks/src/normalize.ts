import type { LandmarkLayoutV1 } from "@signlnk/schemas/landmark_layout_v1";
import { LAYOUT } from "./layout.ts";

/**
 * Landmark normalization for slk-landmarks-v1.
 *
 * Mirrors ml/signlnk_ml/features/normalize.py. Change both together; the golden fixtures in
 * tests/fixtures/normalization are the parity contract (max abs diff <= 1e-5).
 *
 * Per frame: x and y are centred on the neck (midpoint of the two shoulder anchors) and divided by
 * the shoulder width (Euclidean distance in x, y). z is kept unchanged. A frame whose shoulders are
 * missing or closer than MIN_SHOULDER_WIDTH cannot be normalized and becomes all NaN. Math runs in
 * float64 and is rounded to float32 on store, in the same operation order as the Python version.
 */

/** Shoulder widths below this are treated as degenerate (normalized image units). */
export const MIN_SHOULDER_WIDTH = 1e-6;

const at = (values: Float32Array, index: number): number => values[index] ?? Number.NaN;

/** Output-layout indices of the left and right shoulder anchors. */
export function shoulderIndices(layout: LandmarkLayoutV1 = LAYOUT): {
  left: number;
  right: number;
} {
  const left = layout.anchors?.["left_shoulder"];
  const right = layout.anchors?.["right_shoulder"];
  if (left === undefined || right === undefined) {
    throw new Error(`layout ${layout.layout_id} has no shoulder anchors`);
  }
  return { left, right };
}

/** Normalizes one frame (float32 [N * 3]) into a new Float32Array. NaN marks a missing landmark. */
export function normalizeFrame(
  frame: Float32Array,
  layout: LandmarkLayoutV1 = LAYOUT,
): Float32Array {
  const length = layout.n_landmarks * 3;
  if (frame.length !== length) {
    throw new RangeError(
      `expected ${length} values (N=${layout.n_landmarks}), got ${frame.length}`,
    );
  }
  const { left, right } = shoulderIndices(layout);
  const leftX = at(frame, left * 3);
  const leftY = at(frame, left * 3 + 1);
  const rightX = at(frame, right * 3);
  const rightY = at(frame, right * 3 + 1);

  const centerX = (leftX + rightX) / 2;
  const centerY = (leftY + rightY) / 2;
  const deltaX = rightX - leftX;
  const deltaY = rightY - leftY;
  const width = Math.sqrt(deltaX * deltaX + deltaY * deltaY);

  const out = new Float32Array(length);
  const valid =
    Number.isFinite(centerX) &&
    Number.isFinite(centerY) &&
    Number.isFinite(width) &&
    width >= MIN_SHOULDER_WIDTH;
  if (!valid) return out.fill(Number.NaN);

  for (let i = 0; i < length; i += 3) {
    out[i] = (at(frame, i) - centerX) / width;
    out[i + 1] = (at(frame, i + 1) - centerY) / width;
    out[i + 2] = at(frame, i + 2);
  }
  return out;
}

/** Normalizes a clip (float32 [T * N * 3], row-major) frame by frame into a new Float32Array. */
export function normalizeSequence(
  frames: Float32Array,
  layout: LandmarkLayoutV1 = LAYOUT,
): Float32Array {
  const frameLength = layout.n_landmarks * 3;
  if (frames.length % frameLength !== 0) {
    throw new RangeError(`length ${frames.length} is not a multiple of ${frameLength}`);
  }
  const out = new Float32Array(frames.length);
  for (let start = 0; start < frames.length; start += frameLength) {
    out.set(normalizeFrame(frames.subarray(start, start + frameLength), layout), start);
  }
  return out;
}
