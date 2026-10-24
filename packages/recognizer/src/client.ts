import type { FromWorker, InitMessage, LogitsMessage, SignMessage, ToWorker } from "./protocol.ts";

// A script blocked or failed to load fires `error` with an empty message.
const WORKER_FAILED = "the inference worker failed to start (script blocked or failed to load)";

/** A finished sign, with when the main thread got it (same clock as the frame timestamps). */
export interface SignEvent extends Omit<SignMessage, "type"> {
  receivedAt: number;
}

export interface ClientOptions {
  /** `model.onnx` bytes. */
  model: Uint8Array;
  /** Parsed `model.json`. */
  modelJson: unknown;
  /** Same-origin folder holding the onnxruntime-web `.wasm`/`.mjs`, e.g. "/ort/". */
  wasmBase: string;
  decoder: InitMessage["decoder"];
  onSign: (event: SignEvent) => void;
  onError?: (error: Error) => void;
}

/** Main-thread handle to the inference worker. Landmark frames never leave the device. */
export class RecognizerClient {
  /** onnxruntime-web threads in use (1 unless the page is cross-origin isolated). */
  threads = 1;
  private readonly worker: Worker;
  private readonly waiting = new Map<
    number,
    { resolve: (reply: LogitsMessage) => void; reject: (error: Error) => void }
  >();
  private nextId = 0;

  private constructor(worker: Worker) {
    this.worker = worker;
  }

  /** Starts the worker and loads the model; resolves when it is ready to take frames. */
  static create(options: ClientOptions): Promise<RecognizerClient> {
    const worker = new Worker(new URL("./inferenceWorker.ts", import.meta.url));
    const client = new RecognizerClient(worker);
    return new Promise<RecognizerClient>((resolve, reject) => {
      worker.onmessage = (event: MessageEvent<FromWorker>) => {
        const message = event.data;
        if (message.type === "ready") {
          client.threads = message.threads;
          worker.onmessage = (e: MessageEvent<FromWorker>) => client.onMessage(e.data, options);
          worker.onerror = (e) => options.onError?.(new Error(e.message || WORKER_FAILED));
          resolve(client);
        } else if (message.type === "error") {
          worker.terminate();
          reject(new Error(message.message));
        }
      };
      worker.onerror = (event) => {
        worker.terminate();
        reject(new Error(event.message || WORKER_FAILED));
      };
      worker.postMessage({
        type: "init",
        model: options.model,
        modelJson: options.modelJson,
        wasmBase: options.wasmBase,
        decoder: options.decoder,
      } satisfies ToWorker);
    });
  }

  /** Sends one landmark frame (float32 [N * 3], from LandmarkPipeline) captured at `t` ms. */
  pushFrame(t: number, frame: Float32Array): void {
    this.worker.postMessage({ type: "frame", t, frame } satisfies ToWorker);
  }

  /** Runs raw windows through the model as-is (the browser parity check). */
  infer(windows: Float32Array, batch: number): Promise<LogitsMessage> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.waiting.set(id, { resolve, reject });
      this.worker.postMessage({ type: "infer", id, windows, batch } satisfies ToWorker);
    });
  }

  /** Forgets the open sign and the debounce state. */
  reset(): void {
    this.worker.postMessage({ type: "reset" } satisfies ToWorker);
  }

  dispose(): void {
    this.worker.postMessage({ type: "dispose" } satisfies ToWorker);
    this.worker.terminate();
    for (const { reject } of this.waiting.values()) reject(new Error("recognizer disposed"));
    this.waiting.clear();
  }

  private onMessage(message: FromWorker, options: ClientOptions): void {
    if (message.type === "sign") {
      options.onSign({
        caption: message.caption,
        tStart: message.tStart,
        tEnd: message.tEnd,
        tDetected: message.tDetected,
        frameCount: message.frameCount,
        reason: message.reason,
        inferMs: message.inferMs,
        outcome: message.outcome,
        top: message.top,
        otherP: message.otherP,
        receivedAt: performance.now(),
      });
    } else if (message.type === "logits") {
      this.waiting.get(message.id)?.resolve(message);
      this.waiting.delete(message.id);
    } else if (message.type === "error") {
      const error = new Error(message.message);
      for (const { reject } of this.waiting.values()) reject(error); // nobody waits forever
      this.waiting.clear();
      options.onError?.(error);
    }
  }
}
