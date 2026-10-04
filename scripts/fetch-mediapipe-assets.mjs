// Dev-time only: serves MediaPipe from our own origin so the app makes no third-party requests.
//  - copies the WASM fileset from @mediapipe/tasks-vision into apps/web/public/mediapipe/wasm
//  - downloads the three .task models into apps/web/public/mediapipe/models (SHA-256 pinned in
//    scripts/mediapipe-assets.json; a null hash is printed so it can be pinned)
import { createHash } from "node:crypto";
import { cpSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const publicDir = join(root, "apps", "web", "public", "mediapipe");
const manifest = JSON.parse(readFileSync(join(root, "scripts", "mediapipe-assets.json"), "utf8"));
const sha256 = (buffer) => createHash("sha256").update(buffer).digest("hex");

const wasmSource = join(
  root,
  "packages",
  "landmarks",
  "node_modules",
  "@mediapipe",
  "tasks-vision",
  "wasm",
);
if (!existsSync(wasmSource)) {
  console.error("fetch-mediapipe-assets: run `pnpm install` first (tasks-vision WASM not found)");
  process.exit(1);
}
cpSync(wasmSource, join(publicDir, "wasm"), { recursive: true });

const modelsDir = join(publicDir, "models");
mkdirSync(modelsDir, { recursive: true });
let failed = false;
for (const { file, url, sha256: pinned } of manifest.models) {
  const target = join(modelsDir, file);
  if (existsSync(target) && (!pinned || sha256(readFileSync(target)) === pinned)) {
    console.log(`ok       ${file}`);
    continue;
  }
  const response = await fetch(url);
  if (!response.ok) {
    console.error(`FAILED   ${file}: HTTP ${response.status} for ${url}`);
    failed = true;
    continue;
  }
  const buffer = Buffer.from(await response.arrayBuffer());
  const hash = sha256(buffer);
  if (pinned && hash !== pinned) {
    console.error(`MISMATCH ${file}: expected ${pinned}, got ${hash}`);
    failed = true;
    continue;
  }
  writeFileSync(target, buffer);
  const note = pinned ? "" : `  UNPINNED sha256=${hash}`;
  console.log(`fetched  ${file} (${buffer.length} bytes)${note}`);
}
process.exit(failed ? 1 : 0);
