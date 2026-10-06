// Writes tests/fixtures/npy/pattern.npy, the golden file the TS writer and the Python loader share.
// Run: pnpm --filter @signlnk/landmarks write-npy-fixture
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { encodeNpy } from "../src/npy.ts";
import { PATTERN_FRAMES, patternArray } from "../src/npyPattern.ts";

const layoutUrl = new URL("../../schemas/layouts/slk-landmarks-v1.json", import.meta.url);
const landmarkCount = (JSON.parse(readFileSync(layoutUrl, "utf8")) as { n_landmarks: number })
  .n_landmarks;

const target = new URL("../../../tests/fixtures/npy/pattern.npy", import.meta.url);
mkdirSync(new URL(".", target), { recursive: true });
writeFileSync(target, encodeNpy(patternArray(landmarkCount), [PATTERN_FRAMES, landmarkCount, 3]));
console.log("wrote tests/fixtures/npy/pattern.npy");
