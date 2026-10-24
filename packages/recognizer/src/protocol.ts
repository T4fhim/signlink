import type { RecognitionOutputV1 } from "@signlnk/schemas/recognition_output_v1";
import type { DecoderConfig, Outcome } from "./decoder/index.ts";

/**
 * Messages between the main thread and the inference worker. Timestamps (`t`, `tStart`, ...) are
 * on the caller's clock (the main thread's performance.now()); the worker never reads a clock to
 * stamp them, so latency is measured on one clock. Frames are landmark tensors and stay on device.
 */
export interface InitMessage {
  type: "init";
  /** `model.onnx` bytes. */
  model: Uint8Array;
  /** Parsed `model.json`. */
  modelJson: unknown;
  /** Where the onnxruntime-web `.wasm`/`.mjs` are served from (same origin), e.g. "/ort/". */
  wasmBase?: string;
  decoder: Omit<DecoderConfig, "model">;
}
/** One landmark frame (float32 [N * 3], aspect-corrected, not normalized) at `t` ms. */
export interface FrameMessage {
  type: "frame";
  t: number;
  frame: Float32Array;
}
/** Runs windows through the model as-is (the browser parity check). */
export interface InferMessage {
  type: "infer";
  id: number;
  windows: Float32Array;
  batch: number;
}
export interface ResetMessage {
  type: "reset";
}
export interface DisposeMessage {
  type: "dispose";
}
export type ToWorker = InitMessage | FrameMessage | InferMessage | ResetMessage | DisposeMessage;

export interface ReadyMessage {
  type: "ready";
  /** onnxruntime-web threads in use (1 unless the page is cross-origin isolated). */
  threads: number;
}
/** A finished sign; `caption` is null when nothing should be shown (other, low confidence, repeat). */
export interface SignMessage {
  type: "sign";
  caption: RecognitionOutputV1 | null;
  tStart: number;
  tEnd: number;
  tDetected: number;
  frameCount: number;
  /** "max": cut while still moving, so tEnd is not a real sign end (excluded from latency). */
  reason: "quiet" | "max";
  /** Model run time for this sign, ms. */
  inferMs: number;
  /** Why the sign did or did not become a caption (for the report's per-sign log). */
  outcome: Outcome;
  /** Best candidates over the vocabulary, best first. */
  top: { gloss: string; p: number }[];
  /** Probability of `<other>` over the vocabulary plus `<other>`. */
  otherP: number;
}
export interface LogitsMessage {
  type: "logits";
  id: number;
  logits: Float32Array;
  embedding: Float32Array;
  inferMs: number;
}
export interface ErrorMessage {
  type: "error";
  message: string;
}
export type FromWorker = ReadyMessage | SignMessage | LogitsMessage | ErrorMessage;
