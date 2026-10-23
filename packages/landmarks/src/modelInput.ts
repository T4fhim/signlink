import { LANDMARK_COUNT } from "./layout.ts";

/**
 * Model input for the recognizer: the TS twin of `make_input` and `resample_nearest` in
 * ml/signlnk_ml/training/features.py. Change both together; the golden fixture
 * tests/fixtures/model_input/cases.json is the parity contract (max abs diff <= 1e-5).
 *
 * Input is a normalized clip (float32 [T * N * 3], row-major, NaN = missing). A row of the output
 * holds the coordinates of every landmark (x, y, and z with `useZ`; NaN -> 0), then one 1/0
 * "landmark present" flag per landmark. NaN is replaced with 0 here and nowhere else. Frame-to-frame
 * velocity (`velocity` in training) is not ported: the exported model is velocity-free
 * (`model.json` input.velocity = false).
 */

export interface ModelInputOptions {
  /** Include the z coordinate; must equal `model.json` input.use_z. */
  useZ: boolean;
  /** Landmarks per frame; defaults to the layout's N. */
  nLandmarks?: number;
}

/** Width of one output row (coordinates plus a presence flag per landmark). */
export function inputDim(nLandmarks: number, useZ: boolean): number {
  return nLandmarks * (useZ ? 3 : 2) + nLandmarks;
}

/** [T * N * 3] -> [T * F], F = inputDim(N, useZ). Never modifies its input. */
export function makeInput(window: Float32Array, options: ModelInputOptions): Float32Array {
  const n = options.nLandmarks ?? LANDMARK_COUNT;
  const frameLength = n * 3;
  if (window.length % frameLength !== 0) {
    throw new RangeError(`length ${window.length} is not a multiple of ${frameLength} (N=${n})`);
  }
  const used = options.useZ ? 3 : 2;
  const dim = inputDim(n, options.useZ);
  const frames = window.length / frameLength;
  const out = new Float32Array(frames * dim);
  for (let t = 0; t < frames; t++) {
    const source = t * frameLength;
    const row = t * dim;
    for (let landmark = 0; landmark < n; landmark++) {
      let present = 1;
      for (let c = 0; c < used; c++) {
        const value = window[source + landmark * 3 + c] ?? Number.NaN;
        if (Number.isFinite(value)) out[row + landmark * used + c] = value;
        else present = 0;
      }
      out[row + n * used + landmark] = present;
    }
  }
  return out;
}

/**
 * Nearest-frame resample of a clip (float32 [frames * frameLength]) to `length` frames, as the
 * model was trained: index = min(floor(i * frames / length), frames - 1). It both upsamples (frames
 * repeat) and downsamples; NaN is never interpolated.
 */
export function resampleNearest(
  window: Float32Array,
  frameLength: number,
  length: number,
): Float32Array {
  if (window.length % frameLength !== 0) {
    throw new RangeError(`length ${window.length} is not a multiple of ${frameLength}`);
  }
  const frames = window.length / frameLength;
  if (frames === 0) throw new RangeError("cannot resample an empty sequence");
  const out = new Float32Array(length * frameLength);
  for (let i = 0; i < length; i++) {
    const source = Math.min(Math.floor((i * frames) / length), frames - 1);
    out.set(window.subarray(source * frameLength, (source + 1) * frameLength), i * frameLength);
  }
  return out;
}
