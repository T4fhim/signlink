/** Reads the `slk-parity-v1` files written by `signlnk_ml.export` and compares outputs. */

/** PLAN §7 Phase 1 step 5: max abs logit difference between PyTorch and the runtime. */
export const PARITY_TOLERANCE = 1e-4;

export interface Tensor {
  shape: number[];
  data: Float32Array;
}

export interface ParityFile {
  tolerance: number;
  windows: Tensor;
  logits: Tensor;
  embedding: Tensor;
}

interface EncodedTensor {
  shape: number[];
  b64: string;
}

const decodeTensor = (entry: EncodedTensor): Tensor => {
  const bytes = Buffer.from(entry.b64, "base64");
  const count = entry.shape.reduce((a, b) => a * b, 1);
  if (bytes.length !== count * 4) {
    throw new Error(
      `parity tensor data (${bytes.length} bytes) does not match shape [${entry.shape}]`,
    );
  }
  // Copy into an aligned buffer: Buffer slices from the pool are not 4-byte aligned.
  const data = new Float32Array(count);
  new Uint8Array(data.buffer).set(bytes);
  return { shape: entry.shape, data };
};

export const decodeParityFile = (json: unknown): ParityFile => {
  const file = json as {
    format: string;
    tolerance: number;
    windows: EncodedTensor;
    logits: EncodedTensor;
    embedding: EncodedTensor;
  };
  if (file.format !== "slk-parity-v1") throw new Error(`unknown parity format ${file.format}`);
  return {
    tolerance: file.tolerance,
    windows: decodeTensor(file.windows),
    logits: decodeTensor(file.logits),
    embedding: decodeTensor(file.embedding),
  };
};

export const maxAbsDiff = (a: Float32Array, b: Float32Array): number => {
  if (a.length !== b.length) throw new Error(`length mismatch: ${a.length} vs ${b.length}`);
  let worst = 0;
  for (let i = 0; i < a.length; i += 1)
    worst = Math.max(worst, Math.abs((a[i] ?? 0) - (b[i] ?? 0)));
  return worst;
};
