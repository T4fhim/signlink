// Dev-time only: serves onnxruntime-web's WebAssembly from our own origin (the page sets
// ort.env.wasm.wasmPaths = "/ort/"), so inference makes no third-party request.
//  - copies ort-wasm-simd-threaded.{wasm,mjs} from onnxruntime-web into apps/web/public/ort
// Not committed (apps/web/public/ort/ is gitignored). ONNX Runtime's LICENSE and
// ThirdPartyNotices must ship with it in a release (Phase 1 step 7).
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = join(root, "packages", "recognizer", "node_modules", "onnxruntime-web", "dist");
const target = join(root, "apps", "web", "public", "ort");
const files = ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"];

if (!files.every((file) => existsSync(join(source, file)))) {
  console.error("fetch-ort-assets: run `pnpm install` first (onnxruntime-web files not found)");
  process.exit(1);
}
mkdirSync(target, { recursive: true });
for (const file of files) {
  copyFileSync(join(source, file), join(target, file));
  console.log(`copied   ${file}`);
}
