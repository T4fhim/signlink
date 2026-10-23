import * as ort from "onnxruntime-web";
import { createRecognizerWith, type InputShape, type Recognizer } from "./runtime.ts";

export type { InputShape, Recognizer } from "./runtime.ts";

/**
 * Loads an exported recognizer (`model.onnx` bytes) with onnxruntime-web's default entry, which is
 * the Node build: for scripts and tests. Browsers use the inference worker (`onnxruntime-web/wasm`).
 */
export const createRecognizer = (model: Uint8Array, shape: InputShape): Promise<Recognizer> =>
  createRecognizerWith(ort, model, shape);
