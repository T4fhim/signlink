import * as ort from "onnxruntime-web";

/** Shape of one model input window, from `model.json` (`input.length`, `input.dim`). */
export interface InputShape {
  length: number;
  dim: number;
}

export interface Recognizer {
  /** `windows` is `batch * length * dim` floats, row-major. Returns logits and embeddings. */
  run(
    windows: Float32Array,
    batch: number,
  ): Promise<{ logits: Float32Array; embedding: Float32Array }>;
  dispose(): Promise<void>;
}

/** Loads an exported recognizer (`model.onnx` bytes) on onnxruntime-web's WebAssembly backend. */
export const createRecognizer = async (
  model: Uint8Array,
  shape: InputShape,
): Promise<Recognizer> => {
  const session = await ort.InferenceSession.create(model, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  return {
    async run(windows, batch) {
      if (windows.length !== batch * shape.length * shape.dim) {
        throw new Error(
          `input size ${windows.length} does not match ${batch} x ${shape.length} x ${shape.dim}`,
        );
      }
      const input = new ort.Tensor("float32", windows, [batch, shape.length, shape.dim]);
      const out = await session.run({ windows: input });
      const output = (name: string): Float32Array => {
        const tensor = out[name];
        if (!tensor) throw new Error(`model has no output named ${name}`);
        return tensor.data as Float32Array;
      };
      return { logits: output("logits"), embedding: output("embedding") };
    },
    dispose: () => session.release(),
  };
};
