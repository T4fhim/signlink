import type { Delegate, FromWorker, FrameResult, PipelineOptions, ToWorker } from "./protocol.ts";

interface Pending {
  resolve: (result: FrameResult) => void;
  reject: (error: Error) => void;
}

/** Main-thread handle to the landmark Web Worker. Frames never leave the device. */
export class LandmarkPipeline {
  private worker: Worker | null = null;
  private readonly pending = new Map<number, Pending>();
  private nextId = 0;

  /** Starts the worker and loads the three models. Resolves with the delegate actually in use. */
  init(options: PipelineOptions): Promise<Delegate> {
    const worker = new Worker(new URL("./worker.ts", import.meta.url));
    this.worker = worker;
    const full: Required<PipelineOptions> = {
      delegate: "GPU",
      poseEvery: 1,
      faceEvery: 1,
      handAssignment: "wrist",
      swapHandedness: false,
      ...options,
    };
    return new Promise<Delegate>((resolve, reject) => {
      worker.onmessage = (event: MessageEvent<FromWorker>) => {
        const message = event.data;
        if (message.type === "ready") {
          worker.onmessage = (e: MessageEvent<FromWorker>) => this.onMessage(e.data);
          resolve(message.delegate);
        } else if (message.type === "error") {
          reject(new Error(message.message));
        }
      };
      worker.onerror = (event) => reject(new Error(event.message));
      worker.postMessage({ type: "init", options: full } satisfies ToWorker);
    });
  }

  /** Extracts one frame's landmarks. Call again only after the returned promise settles. */
  async process(source: ImageBitmapSource, t: number): Promise<FrameResult> {
    const worker = this.worker;
    if (!worker) throw new Error("LandmarkPipeline.init() has not completed");
    const bitmap = await createImageBitmap(source);
    const id = this.nextId++;
    return new Promise<FrameResult>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      worker.postMessage({ type: "frame", id, t, bitmap } satisfies ToWorker, [bitmap]);
    });
  }

  dispose(): void {
    this.worker?.terminate();
    this.worker = null;
    for (const { reject } of this.pending.values()) reject(new Error("pipeline disposed"));
    this.pending.clear();
  }

  private onMessage(message: FromWorker): void {
    if (message.type === "result") {
      const { t, frame, present, timingsMs } = message;
      this.pending.get(message.id)?.resolve({ t, frame, present, timingsMs });
      this.pending.delete(message.id);
    } else if (message.type === "error" && message.id !== null) {
      this.pending.get(message.id)?.reject(new Error(message.message));
      this.pending.delete(message.id);
    }
  }
}
