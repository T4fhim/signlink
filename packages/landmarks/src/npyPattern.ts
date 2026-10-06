/** Frames in the golden `.npy` pattern. */
export const PATTERN_FRAMES = 4;

/**
 * A deterministic [PATTERN_FRAMES, N, 3] array, identical to the one built in
 * ml/tests/test_recordings.py: value = t * 1000 + landmark + 0.25 * coordinate, with NaN wherever
 * (t + landmark + coordinate) % 17 == 0. All values are exact in float32.
 */
export function patternArray(landmarkCount: number): Float32Array {
  const out = new Float32Array(PATTERN_FRAMES * landmarkCount * 3);
  let i = 0;
  for (let t = 0; t < PATTERN_FRAMES; t++) {
    for (let k = 0; k < landmarkCount; k++) {
      for (let c = 0; c < 3; c++) {
        out[i++] = (t + k + c) % 17 === 0 ? Number.NaN : t * 1000 + k + 0.25 * c;
      }
    }
  }
  return out;
}
