// Writes packages/schemas/layouts/slk-landmarks-v1.json from the MediaPipe face connection sets.
// Run: pnpm --filter @signlnk/landmarks write-layout
import { writeFileSync } from "node:fs";
import { FaceLandmarker } from "@mediapipe/tasks-vision";
import { buildLayout } from "../src/layoutSpec.ts";

const layout = buildLayout({
  lips: FaceLandmarker.FACE_LANDMARKS_LIPS,
  leftEyebrow: FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
  rightEyebrow: FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
  leftEye: FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
  rightEye: FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
});

const target = new URL("../../schemas/layouts/slk-landmarks-v1.json", import.meta.url);
writeFileSync(target, `${JSON.stringify(layout, null, 2)}\n`);
console.log(`wrote ${layout.layout_id}: N=${layout.n_landmarks}`);
