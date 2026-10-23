// Stages a trained recognizer for the web app (apps/web/public/model/, gitignored).
//   node scripts/stage-model.mjs <run-dir> [--vocab ml/vocab/<list>.yaml]
// <run-dir> holds model.onnx and model.json from `signlnk_ml.export.onnx_export` (for example
// %SIGNLNK_DATA_DIR%\models\baseline-seed1). Copies model.onnx, writes model.json without
// source.command (it can contain local paths), and with --vocab writes vocabulary.json (the
// list's sign names, in order). Refuses a model that is not release-track (CLAUDE.md rule 4).
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const target = join(root, "apps", "web", "public", "model");
const args = process.argv.slice(2);
const vocabAt = args.indexOf("--vocab");
const vocabFile = vocabAt === -1 ? null : args[vocabAt + 1];
const sizeAt = args.indexOf("--size");
const size = sizeAt === -1 ? null : Number(args[sizeAt + 1]);
const optionValues = new Set([vocabAt, sizeAt].filter((at) => at !== -1).map((at) => at + 1));
const runDir = args.find((arg, i) => !arg.startsWith("--") && !optionValues.has(i));

const fail = (message) => {
  console.error(`stage-model: ${message}`);
  process.exit(1);
};
if (
  !runDir ||
  (vocabAt !== -1 && !vocabFile) ||
  (sizeAt !== -1 && !(Number.isInteger(size) && size > 0))
) {
  fail("usage: node scripts/stage-model.mjs <run-dir> [--vocab <list.yaml> [--size N]]");
}
const onnx = join(runDir, "model.onnx");
const jsonPath = join(runDir, "model.json");
if (!existsSync(onnx) || !existsSync(jsonPath)) fail(`${runDir} needs model.onnx and model.json`);

const info = JSON.parse(readFileSync(jsonPath, "utf8"));
if (info.track !== "release") fail(`track is "${info.track}", only release-track models ship`);
if (info.format !== "slk-recognizer-v1") fail(`unsupported format ${info.format}`);
if (info.source) delete info.source.command;

mkdirSync(target, { recursive: true });
copyFileSync(onnx, join(target, "model.onnx"));
writeFileSync(join(target, "model.json"), `${JSON.stringify(info, null, 2)}\n`);
console.log(`staged   model.onnx + model.json (${info.classes.length} classes)`);
// parity.json (PyTorch outputs for fixed windows) lets /dev/parity re-check the model in a browser.
if (existsSync(join(runDir, "parity.json"))) {
  copyFileSync(join(runDir, "parity.json"), join(target, "parity.json"));
  console.log("staged   parity.json");
}

if (vocabFile) {
  // The vocabulary files are a flat `signs:` list; one name per `- name` line, `#` comments.
  const text = readFileSync(resolve(root, vocabFile), "utf8");
  const body = text.slice(text.search(/^signs:\s*$/m));
  // --size N keeps the first N (the lists are in priority order), like `train --set vocabulary.size`.
  const listed = [...body.matchAll(/^\s*-\s*"?([^"#\s]+)"?\s*(?:#.*)?$/gm)].map((m) => m[1]);
  const signs = size === null ? listed : listed.slice(0, size);
  const unknown = signs.filter((sign) => !info.classes.includes(sign));
  if (signs.length === 0 || unknown.length > 0) fail(`bad vocabulary (unknown: ${unknown})`);
  writeFileSync(join(target, "vocabulary.json"), `${JSON.stringify(signs, null, 2)}\n`);
  console.log(`staged   vocabulary.json (${signs.length} signs)`);
}
