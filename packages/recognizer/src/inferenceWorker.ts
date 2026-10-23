// Browser entry: onnxruntime-web/wasm (the default browser entry pulls in the untested jsep build).
import * as ort from "onnxruntime-web/wasm";
import { InferenceHost } from "./host.ts";
import type { FromWorker, ToWorker } from "./protocol.ts";

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const host = new InferenceHost(ort, (message: FromWorker) => ctx.postMessage(message));

ctx.onmessage = (event: MessageEvent<ToWorker>) => {
  void host.handle(event.data);
};
