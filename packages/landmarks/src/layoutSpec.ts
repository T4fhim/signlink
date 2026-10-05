import type { LandmarkLayoutV1 } from "@signlnk/schemas/landmark_layout_v1";

export interface Connection {
  start: number;
  end: number;
}

/** MediaPipe Face Landmarker connection sets (FaceLandmarker.FACE_LANDMARKS_*). */
export interface FaceConnectionSets {
  lips: Connection[];
  leftEyebrow: Connection[];
  rightEyebrow: Connection[];
  leftEye: Connection[];
  rightEye: Connection[];
}

/** BlazePose indices kept for the upper body, in output order. Names are anatomical (subject's own side). */
export const POSE_SUBSET = [
  ["nose", 0],
  ["left_eye", 2],
  ["right_eye", 5],
  ["left_ear", 7],
  ["right_ear", 8],
  ["left_shoulder", 11],
  ["right_shoulder", 12],
  ["left_elbow", 13],
  ["right_elbow", 14],
  ["left_wrist", 15],
  ["right_wrist", 16],
] as const;

/** Face mesh nose tip. */
export const NOSE_TIP_INDEX = 1;

const HAND_LANDMARK_COUNT = 21;

const uniqueSorted = (connections: Connection[]): number[] =>
  [...new Set(connections.flatMap((c) => [c.start, c.end]))].sort((a, b) => a - b);

/** Builds the slk-landmarks-v1 layout. The face subset comes from the library's own connection sets. */
export function buildLayout(face: FaceConnectionSets): LandmarkLayoutV1 {
  const handIndices = Array.from({ length: HAND_LANDMARK_COUNT }, (_, i) => i);
  const groups: LandmarkLayoutV1["groups"] = [
    { name: "hand_left", source: "hand_left", indices: handIndices },
    { name: "hand_right", source: "hand_right", indices: [...handIndices] },
    { name: "pose", source: "pose", indices: POSE_SUBSET.map(([, index]) => index) },
    { name: "face_lips", source: "face", indices: uniqueSorted(face.lips) },
    { name: "face_left_eyebrow", source: "face", indices: uniqueSorted(face.leftEyebrow) },
    { name: "face_right_eyebrow", source: "face", indices: uniqueSorted(face.rightEyebrow) },
    { name: "face_left_eye", source: "face", indices: uniqueSorted(face.leftEye) },
    { name: "face_right_eye", source: "face", indices: uniqueSorted(face.rightEye) },
    { name: "face_nose_tip", source: "face", indices: [NOSE_TIP_INDEX] },
  ];

  // Identity mapping to the legacy Holistic layout (Kaggle ISLR): hand, BlazePose and face-mesh
  // numbering are shared. Checked on 116 Kaggle sequences (ADR-0006); the Tasks side is [VERIFY]
  // with a real browser recording in the train/serve parity test.
  for (const group of groups) group.legacy_holistic_indices = [...group.indices];

  const poseOffset = groups
    .slice(
      0,
      groups.findIndex((g) => g.name === "pose"),
    )
    .reduce((sum, g) => sum + g.indices.length, 0);
  const poseAnchor = (name: string): number =>
    poseOffset + POSE_SUBSET.findIndex(([n]) => n === name);

  return {
    layout_id: "slk-landmarks-v1",
    n_landmarks: groups.reduce((sum, g) => sum + g.indices.length, 0),
    groups,
    anchors: {
      left_shoulder: poseAnchor("left_shoulder"),
      right_shoulder: poseAnchor("right_shoulder"),
    },
  };
}
