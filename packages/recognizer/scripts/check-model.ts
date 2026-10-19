// Checks an exported model folder (model.onnx, model.json, parity.json) with onnxruntime-web and
// prints the largest differences from PyTorch's outputs.
//   pnpm --filter @signlnk/recognizer check-model <folder>
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { decodeParityFile, maxAbsDiff, PARITY_TOLERANCE } from "../src/parity.ts";
import { createRecognizer } from "../src/session.ts";

const folder = process.argv[2];
if (!folder) {
  console.error("usage: check-model <folder with model.onnx, model.json, parity.json>");
  process.exit(2);
}
const meta = JSON.parse(readFileSync(join(folder, "model.json"), "utf8")) as {
  classes: string[];
  input: { length: number; dim: number };
};
const parity = decodeParityFile(JSON.parse(readFileSync(join(folder, "parity.json"), "utf8")));
const recognizer = await createRecognizer(
  new Uint8Array(readFileSync(join(folder, "model.onnx"))),
  meta.input,
);
const batch = parity.windows.shape[0] as number;
const started = performance.now();
const out = await recognizer.run(parity.windows.data, batch);
const ms = performance.now() - started;
const logits = maxAbsDiff(out.logits, parity.logits.data);
const embedding = maxAbsDiff(out.embedding, parity.embedding.data);
await recognizer.dispose();
console.log(
  `onnxruntime-web (wasm, node) on ${batch} windows, ${meta.classes.length} classes: ` +
    `max |logit diff| ${logits.toExponential(2)}, max |embedding diff| ${embedding.toExponential(2)} ` +
    `(tolerance ${PARITY_TOLERANCE}); first run ${ms.toFixed(0)} ms for the batch`,
);
process.exit(logits <= PARITY_TOLERANCE && embedding <= PARITY_TOLERANCE ? 0 : 1);
