import { Decoder, parseModelInfo } from "./decoder/index.ts";
import type { FromWorker, ToWorker } from "./protocol.ts";
import { createRecognizerWith, type OrtRuntime, type Recognizer } from "./runtime.ts";

/**
 * The inference worker's logic, with no worker globals so it runs the same in Node tests and in
 * the browser worker. Messages are handled one at a time in arrival order (a sign's model run
 * finishes before the next frame is looked at); errors are posted, never thrown.
 */
export class InferenceHost {
  private readonly ort: OrtRuntime;
  private readonly post: (message: FromWorker) => void;
  private recognizer: Recognizer | null = null;
  private decoder: Decoder | null = null;
  private queue: Promise<void> = Promise.resolve();

  constructor(ort: OrtRuntime, post: (message: FromWorker) => void) {
    this.ort = ort;
    this.post = post;
  }

  /** Queues a message; resolves when it has been handled. Never rejects. */
  handle(message: ToWorker): Promise<void> {
    const done = this.queue.then(() => this.process(message));
    this.queue = done;
    return done;
  }

  private async process(message: ToWorker): Promise<void> {
    try {
      switch (message.type) {
        case "init":
          return await this.init(message);
        case "frame":
          return await this.frame(message.t, message.frame);
        case "infer":
          return await this.infer(message.id, message.windows, message.batch);
        case "reset":
          this.decoder?.reset();
          return;
        case "dispose":
          await this.recognizer?.dispose();
          this.recognizer = null;
          this.decoder = null;
          return;
      }
    } catch (error) {
      this.post({ type: "error", message: error instanceof Error ? error.message : String(error) });
    }
  }

  private async init(message: Extract<ToWorker, { type: "init" }>): Promise<void> {
    const info = parseModelInfo(message.modelJson);
    await this.recognizer?.dispose();
    this.recognizer = null;
    this.decoder = null;
    if (message.wasmBase !== undefined) this.ort.env.wasm.wasmPaths = message.wasmBase;
    const recognizer = await createRecognizerWith(this.ort, message.model, {
      length: info.length,
      dim: info.dim,
    });
    this.decoder = new Decoder({ ...message.decoder, model: info });
    this.recognizer = recognizer;
    this.post({ type: "ready", threads: this.ort.env.wasm.numThreads ?? 1 });
  }

  private ready(): { recognizer: Recognizer; decoder: Decoder } {
    if (!this.recognizer || !this.decoder)
      throw new Error("inference host: init has not completed");
    return { recognizer: this.recognizer, decoder: this.decoder };
  }

  private async frame(t: number, frame: Float32Array): Promise<void> {
    const { recognizer, decoder } = this.ready();
    const sign = decoder.push(t, frame);
    if (!sign) return;
    const started = performance.now();
    const { logits } = await recognizer.run(sign.input, 1);
    const inferMs = performance.now() - started;
    this.post({
      type: "sign",
      caption: decoder.interpret(sign, logits),
      tStart: sign.tStart,
      tEnd: sign.tEnd,
      tDetected: sign.tDetected,
      frameCount: sign.frameCount,
      reason: sign.reason,
      inferMs,
    });
  }

  private async infer(id: number, windows: Float32Array, batch: number): Promise<void> {
    const { recognizer } = this.ready();
    const started = performance.now();
    const { logits, embedding } = await recognizer.run(windows, batch);
    this.post({ type: "logits", id, logits, embedding, inferMs: performance.now() - started });
  }
}
