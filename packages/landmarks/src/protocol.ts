import type { Presence } from "./frame.ts";

export type Delegate = "GPU" | "CPU";

export interface PipelineOptions {
  /** Directory serving the MediaPipe WASM fileset. */
  wasmBase: string;
  /** Directory serving hand_landmarker.task, pose_landmarker.task, face_landmarker.task. */
  modelBase: string;
  /** Preferred delegate; falls back to CPU if GPU creation fails. Default GPU. */
  delegate?: Delegate;
  /** Run pose on every Nth frame, reusing the last result in between. Default 1. */
  poseEvery?: number;
  /** Run face on every Nth frame, reusing the last result in between. Default 1. */
  faceEvery?: number;
  /** See AssembleOptions.swapHandedness. Default false. */
  swapHandedness?: boolean;
}

export interface StageTimings {
  hand: number;
  pose: number;
  face: number;
  /** Everything inside the worker for this frame, ms. */
  total: number;
}

export interface FrameResult {
  t: number;
  /** float32 [N * 3] following slk-landmarks-v1; NaN = missing. */
  frame: Float32Array;
  present: Presence;
  timingsMs: StageTimings;
}

export type ToWorker =
  | { type: "init"; options: Required<PipelineOptions> }
  | { type: "frame"; id: number; t: number; bitmap: ImageBitmap };

export type FromWorker =
  | { type: "ready"; delegate: Delegate }
  | { type: "error"; id: number | null; message: string }
  | ({ type: "result"; id: number } & FrameResult);
