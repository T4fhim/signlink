"use client";

import {
  FRAME_LENGTH,
  LANDMARK_COUNT,
  Recorder,
  encodeNpy,
  type Delegate,
  type FrameResult,
  type RecordingSummary,
} from "@signlnk/landmarks";
import { useCallback, useEffect, useRef, useState } from "react";
import { useLandmarkStream } from "../useLandmarkStream";

const RECORD_MS = 3000;
const COUNTDOWN_SECONDS = 3;

type Phase = "idle" | "countdown" | "recording" | "done";

interface Take {
  data: Float32Array;
  summary: RecordingSummary;
}

const fmt = (value: number, digits = 1): string =>
  Number.isFinite(value) ? value.toFixed(digits) : "–";
const pct = (fraction: number): string => `${Math.round(fraction * 100)}%`;
const mark = (on: boolean | undefined): string => (on ? "✔" : "✘");

function fileName(): string {
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace("T", "-").slice(0, 15);
  return `recording-${stamp}.npy`;
}

export default function RecordPage() {
  const recorderRef = useRef<Recorder | null>(null);
  const phaseRef = useRef<Phase>("idle");
  const [phase, setPhaseState] = useState<Phase>("idle");
  const [countdown, setCountdown] = useState(0);
  const [progress, setProgress] = useState(0);
  const [take, setTake] = useState<Take | null>(null);
  const [delegate, setDelegate] = useState<Delegate>("GPU");
  const [poseEvery, setPoseEvery] = useState(2);
  const [faceEvery, setFaceEvery] = useState(2);

  const setPhase = useCallback((next: Phase) => {
    phaseRef.current = next;
    setPhaseState(next);
  }, []);

  const onFrame = useCallback(
    (result: FrameResult) => {
      const recorder = recorderRef.current;
      if (phaseRef.current !== "recording" || !recorder) return;
      const complete = recorder.add(result);
      setProgress(Math.min(1, recorder.summary().durationMs / RECORD_MS));
      if (complete) {
        setTake({ data: recorder.toFloat32(), summary: recorder.summary() });
        recorderRef.current = null;
        setPhase("done");
      }
    },
    [setPhase],
  );

  const { videoRef, status, message, usedDelegate, fps, present, start, stop } =
    useLandmarkStream(onFrame);

  const running = status === "running";

  useEffect(() => {
    if (phase !== "countdown") return;
    if (countdown <= 0) {
      recorderRef.current = new Recorder(FRAME_LENGTH, RECORD_MS);
      setProgress(0);
      setPhase("recording");
      return;
    }
    const timer = setTimeout(() => setCountdown((c) => c - 1), 1000);
    return () => clearTimeout(timer);
  }, [phase, countdown, setPhase]);

  const record = () => {
    setTake(null);
    setCountdown(COUNTDOWN_SECONDS);
    setPhase("countdown");
  };

  const discard = () => {
    recorderRef.current = null;
    setTake(null);
    setPhase("idle");
  };

  const stopCamera = () => {
    discard();
    stop();
  };

  const download = () => {
    if (!take) return;
    const bytes = encodeNpy(take.data, [take.summary.frames, LANDMARK_COUNT, 3]);
    const blob = new Blob([bytes.buffer as ArrayBuffer], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = fileName();
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main
      style={{
        fontFamily: "system-ui, sans-serif",
        maxWidth: 720,
        margin: "1rem auto",
        padding: "0 1rem",
      }}
    >
      <h1>Record landmarks</h1>
      <p>
        Camera video is processed in this browser tab only. A recording holds landmark numbers (not
        video), stays in this tab until you click Download, and is never uploaded. Landmarks include
        your face shape, so treat the file as personal data.
      </p>

      <video
        ref={videoRef}
        muted
        playsInline
        style={{ width: "100%", maxWidth: 640, background: "#111", transform: "scaleX(-1)" }}
      />
      {running && (
        <p role="status" style={{ color: "#b00020" }}>
          ● Camera on
        </p>
      )}

      <fieldset disabled={status === "loading" || running}>
        <legend>Settings</legend>
        <label>
          Delegate{" "}
          <select value={delegate} onChange={(e) => setDelegate(e.target.value as Delegate)}>
            <option value="GPU">GPU</option>
            <option value="CPU">CPU</option>
          </select>
        </label>{" "}
        <label>
          Pose every{" "}
          <select value={poseEvery} onChange={(e) => setPoseEvery(Number(e.target.value))}>
            <option value={1}>frame</option>
            <option value={2}>2nd frame</option>
          </select>
        </label>{" "}
        <label>
          Face every{" "}
          <select value={faceEvery} onChange={(e) => setFaceEvery(Number(e.target.value))}>
            <option value={1}>frame</option>
            <option value={2}>2nd frame</option>
          </select>
        </label>
      </fieldset>

      <p>
        {running ? (
          <button onClick={stopCamera}>Stop camera</button>
        ) : (
          <button
            onClick={() => void start({ delegate, poseEvery, faceEvery })}
            disabled={status === "loading"}
          >
            Start camera
          </button>
        )}{" "}
        <button
          onClick={record}
          disabled={!running || phase === "countdown" || phase === "recording"}
        >
          Record {RECORD_MS / 1000} seconds
        </button>{" "}
        <button onClick={download} disabled={!take}>
          Download .npy
        </button>{" "}
        <button onClick={discard} disabled={!take}>
          Discard
        </button>
      </p>

      <p role="status" aria-live="polite">
        {message}
        {phase === "countdown" && `Get ready… ${countdown}`}
        {phase === "recording" && `Recording… ${pct(progress)}`}
        {phase === "done" && "Recorded. Download it or discard it."}
      </p>

      {running && (
        <section aria-label="Live">
          <p>
            <strong>{fmt(fps)} fps</strong> · delegate {usedDelegate ?? "–"} · left hand{" "}
            {mark(present?.leftHand)} · right hand {mark(present?.rightHand)} · pose{" "}
            {mark(present?.pose)} · face {mark(present?.face)}
          </p>
        </section>
      )}

      {take && (
        <section aria-label="Recording summary">
          <h2>Last recording</h2>
          <p>
            {take.summary.frames} frames over {fmt(take.summary.durationMs / 1000, 2)} s (
            {fmt(take.summary.fps)} fps), shape [{take.summary.frames}, {LANDMARK_COUNT}, 3],
            slk-landmarks-v1 landmarks (y in the reference aspect, not yet normalized).
          </p>
          <p>
            Detected in: left hand {pct(take.summary.presence.leftHand)} · right hand{" "}
            {pct(take.summary.presence.rightHand)} · pose {pct(take.summary.presence.pose)} · face{" "}
            {pct(take.summary.presence.face)}
          </p>
          <p>
            Save it in <code>$SIGNLNK_DATA_DIR/serve-recordings/</code> so the parity test can read
            it.
          </p>
        </section>
      )}
    </main>
  );
}
