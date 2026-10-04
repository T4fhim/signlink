/* AUTO-GENERATED from packages/schemas/landmark_layout.v1.json. Do not edit; run `pnpm gen:types`. */

/**
 * Shape of a landmark layout document (which source landmarks make up the [T, N, 3] tensor). The concrete slk-landmarks-v1 index subset is frozen in Phase 0 step 3. Add fields only.
 */
export interface LandmarkLayoutV1 {
  layout_id: string;
  /**
   * N in the float32 [T, N, 3] tensor.
   */
  n_landmarks: number;
  /**
   * Concatenated in order to form the N landmarks.
   *
   * @minItems 1
   */
  groups: [Group, ...Group[]];
  /**
   * Named indices into the output layout, e.g. left_shoulder and right_shoulder for normalization.
   */
  anchors?: {
    [k: string]: number;
  };
  [k: string]: unknown;
}
export interface Group {
  name: string;
  source: "hand_left" | "hand_right" | "pose" | "face";
  /**
   * Landmark indices in the MediaPipe Tasks output, in output order.
   */
  indices: number[];
  /**
   * Matching indices in the legacy Holistic layout (Kaggle ISLR); null where there is no counterpart. Same length as indices.
   */
  legacy_holistic_indices?: (number | null)[];
  [k: string]: unknown;
}
