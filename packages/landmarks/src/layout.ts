import layoutJson from "@signlnk/schemas/layouts/slk-landmarks-v1.json";
import type { LandmarkLayoutV1 } from "@signlnk/schemas/landmark_layout_v1";

/** The frozen slk-landmarks-v1 layout (single source: packages/schemas/layouts). */
export const LAYOUT = layoutJson as unknown as LandmarkLayoutV1;

export const LANDMARK_LAYOUT_ID = LAYOUT.layout_id;

/** N in the float32 [T, N, 3] tensor. */
export const LANDMARK_COUNT = LAYOUT.n_landmarks;

/** Floats per frame (x, y, z per landmark). */
export const FRAME_LENGTH = LANDMARK_COUNT * 3;
