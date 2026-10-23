import type * as Ort from "onnxruntime-web";

/** The onnxruntime-web module a caller chose: the default entry in Node, `/wasm` in browsers. */
export type OrtRuntime = typeof Ort;

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

/**
 * Loads an exported recognizer (`model.onnx` bytes) on the WebAssembly backend of the given
 * onnxruntime-web module. The caller picks the entry point: a browser must import
 * `onnxruntime-web/wasm` (the default browser entry pulls in the untested jsep build).
 */
export const createRecognizerWith = async (
  ort: OrtRuntime,
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
