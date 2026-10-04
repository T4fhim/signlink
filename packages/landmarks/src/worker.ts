import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";
import { assembleFrame, type Point, type RawDetections } from "./frame.ts";
import { LAYOUT } from "./layout.ts";
import type { Delegate, FromWorker, StageTimings, ToWorker } from "./protocol.ts";

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const send = (message: FromWorker, transfer: Transferable[] = []): void =>
  ctx.postMessage(message, transfer);

interface State {
  hand: HandLandmarker;
  pose: PoseLandmarker;
  face: FaceLandmarker;
  options: Extract<ToWorker, { type: "init" }>["options"];
  frameIndex: number;
  lastTimestamp: number;
  lastPose: readonly Point[] | null;
  lastFace: readonly Point[] | null;
}
let state: State | null = null;

async function createLandmarkers(
  options: State["options"],
  delegate: Delegate,
): Promise<Pick<State, "hand" | "pose" | "face">> {
  const vision = await FilesetResolver.forVisionTasks(options.wasmBase);
  const base = (file: string) => ({
    modelAssetPath: `${options.modelBase}/${file}`,
    delegate,
  });
  const [hand, pose, face] = await Promise.all([
    HandLandmarker.createFromOptions(vision, {
      baseOptions: base("hand_landmarker.task"),
      runningMode: "VIDEO",
      numHands: 2,
    }),
    PoseLandmarker.createFromOptions(vision, {
      baseOptions: base("pose_landmarker.task"),
      runningMode: "VIDEO",
      numPoses: 1,
    }),
    FaceLandmarker.createFromOptions(vision, {
      baseOptions: base("face_landmarker.task"),
      runningMode: "VIDEO",
      numFaces: 1,
    }),
  ]);
  return { hand, pose, face };
}

async function init(options: State["options"]): Promise<void> {
  let delegate = options.delegate;
  let landmarkers: Awaited<ReturnType<typeof createLandmarkers>>;
  try {
    landmarkers = await createLandmarkers(options, delegate);
  } catch (error) {
    if (delegate === "CPU") throw error;
    delegate = "CPU";
    landmarkers = await createLandmarkers(options, delegate);
  }
  state = {
    ...landmarkers,
    options,
    frameIndex: 0,
    lastTimestamp: 0,
    lastPose: null,
    lastFace: null,
  };
  send({ type: "ready", delegate });
}

function processFrame(id: number, t: number, bitmap: ImageBitmap): void {
  if (!state) throw new Error("worker not initialised");
  const s = state;
  const started = performance.now();
  // MediaPipe requires strictly increasing integer timestamps per landmarker.
  const timestamp = Math.max(Math.round(t), s.lastTimestamp + 1);
  s.lastTimestamp = timestamp;

  const handResult = s.hand.detectForVideo(bitmap, timestamp);
  const afterHand = performance.now();

  if (s.frameIndex % s.options.poseEvery === 0) {
    s.lastPose = s.pose.detectForVideo(bitmap, timestamp).landmarks[0] ?? null;
  }
  const afterPose = performance.now();

  if (s.frameIndex % s.options.faceEvery === 0) {
    s.lastFace = s.face.detectForVideo(bitmap, timestamp).faceLandmarks[0] ?? null;
  }
  const afterFace = performance.now();
  s.frameIndex += 1;
  bitmap.close();

  const raw: RawDetections = {
    hands: handResult.landmarks.map((points, i) => ({
      label: handResult.handedness[i]?.[0]?.categoryName === "Left" ? "Left" : "Right",
      points,
    })),
    pose: s.lastPose,
    face: s.lastFace,
  };
  const { frame, present } = assembleFrame(raw, LAYOUT, {
    swapHandedness: s.options.swapHandedness,
  });
  const timingsMs: StageTimings = {
    hand: afterHand - started,
    pose: afterPose - afterHand,
    face: afterFace - afterPose,
    total: performance.now() - started,
  };
  send({ type: "result", id, t, frame, present, timingsMs }, [frame.buffer]);
}

ctx.onmessage = (event: MessageEvent<ToWorker>) => {
  const message = event.data;
  if (message.type === "init") {
    init(message.options).catch((error: unknown) =>
      send({ type: "error", id: null, message: String(error) }),
    );
    return;
  }
  try {
    processFrame(message.id, message.t, message.bitmap);
  } catch (error) {
    message.bitmap.close();
    send({ type: "error", id: message.id, message: String(error) });
  }
};
