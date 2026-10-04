import { FaceLandmarker } from "@mediapipe/tasks-vision";
import { describe, expect, it } from "vitest";
import { FRAME_LENGTH, LANDMARK_COUNT, LAYOUT } from "./layout.ts";
import { buildLayout } from "./layoutSpec.ts";

const FACE_MESH_POINTS = 468; // indices 468+ are iris points, which the layout excludes
const POSE_POINTS = 33;
const HAND_POINTS = 21;

describe("slk-landmarks-v1 layout", () => {
  it("matches what the MediaPipe face connection sets derive (run `write-layout` if this fails)", () => {
    const derived = buildLayout({
      lips: FaceLandmarker.FACE_LANDMARKS_LIPS,
      leftEyebrow: FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
      rightEyebrow: FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
      leftEye: FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
      rightEye: FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
    });
    expect(LAYOUT).toEqual(derived);
  });

  it("has N = 146 and a 438-float frame", () => {
    expect(LANDMARK_COUNT).toBe(146);
    expect(FRAME_LENGTH).toBe(438);
    const total = LAYOUT.groups.reduce((sum, g) => sum + g.indices.length, 0);
    expect(total).toBe(LANDMARK_COUNT);
  });

  it("keeps indices unique and inside each source's range", () => {
    const limit = {
      hand_left: HAND_POINTS,
      hand_right: HAND_POINTS,
      pose: POSE_POINTS,
      face: FACE_MESH_POINTS,
    };
    for (const group of LAYOUT.groups) {
      expect(new Set(group.indices).size, group.name).toBe(group.indices.length);
      for (const index of group.indices) {
        expect(index, group.name).toBeGreaterThanOrEqual(0);
        expect(index, group.name).toBeLessThan(limit[group.source]);
      }
    }
  });

  it("anchors the shoulders at the pose landmarks 11 and 12", () => {
    const flat: { source: string; index: number }[] = [];
    for (const group of LAYOUT.groups) {
      for (const index of group.indices) flat.push({ source: group.source, index });
    }
    expect(flat.length).toBe(LANDMARK_COUNT);
    const anchors = LAYOUT.anchors ?? {};
    expect(flat[anchors["left_shoulder"] ?? -1]).toEqual({ source: "pose", index: 11 });
    expect(flat[anchors["right_shoulder"] ?? -1]).toEqual({ source: "pose", index: 12 });
  });
});
