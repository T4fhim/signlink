"use client";

import {
  LandmarkPipeline,
  RollingWindow,
  fpsFromTimestamps,
  type Delegate,
  type FrameResult,
} from "@signlnk/landmarks";
import { useCallback, useEffect, useRef, useState } from "react";

const TARGET_FPS = 25;
const WINDOW = 120;
const MIN_FRAMES_FOR_VERDICT = 60;

interface Stats {
  frames: number;
  fps: number;
  endToEndMs: { p50: number; p95: number };
  stageMs: Record<"hand" | "pose" | "face" | "total", { p50: number; p95: number }>;
  present: FrameResult["present"] | null;
}

const fmt = (value: number, digits = 1): string =>
  Number.isFinite(value) ? value.toFixed(digits) : "–";

export default function LandmarksBenchmark() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const pipelineRef = useRef<LandmarkPipeline | null>(null);
  const runningRef = useRef(false);
  const streamRef = useRef<MediaStream | null>(null);
  const samples = useRef({
    done: [] as number[],
    endToEnd: new RollingWindow(WINDOW),
    hand: new RollingWindow(WINDOW),
    pose: new RollingWindow(WINDOW),
    face: new RollingWindow(WINDOW),
    total: new RollingWindow(WINDOW),
    frames: 0,
    present: null as FrameResult["present"] | null,
  });

  const [status, setStatus] = useState<"idle" | "loading" | "running" | "error">("idle");
  const [message, setMessage] = useState("");
  const [delegate, setDelegate] = useState<Delegate>("GPU");
  const [usedDelegate, setUsedDelegate] = useState<Delegate | null>(null);
  const [poseEvery, setPoseEvery] = useState(2);
  const [faceEvery, setFaceEvery] = useState(2);
  const [stats, setStats] = useState<Stats | null>(null);

  const snapshot = useCallback((): Stats => {
    const s = samples.current;
    const q = (w: RollingWindow) => ({ p50: w.median, p95: w.p95 });
    return {
      frames: s.frames,
      fps: fpsFromTimestamps(s.done),
      endToEndMs: q(s.endToEnd),
      stageMs: { hand: q(s.hand), pose: q(s.pose), face: q(s.face), total: q(s.total) },
      present: s.present,
    };
  }, []);

  const stop = useCallback(() => {
    runningRef.current = false;
    pipelineRef.current?.dispose();
    pipelineRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setStatus("idle");
  }, []);

  useEffect(() => stop, [stop]);

  useEffect(() => {
    if (status !== "running") return;
    const timer = setInterval(() => setStats(snapshot()), 500);
    return () => clearInterval(timer);
  }, [status, snapshot]);

  const start = async () => {
    const s = samples.current;
    s.done = [];
    s.frames = 0;
    s.present = null;
    for (const w of [s.endToEnd, s.hand, s.pose, s.face, s.total]) w.clear();
    setStatus("loading");
    setMessage("Loading models and camera…");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, frameRate: 30 },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error("video element missing");
      video.srcObject = stream;
      await video.play();

      const pipeline = new LandmarkPipeline();
      pipelineRef.current = pipeline;
      setUsedDelegate(
        await pipeline.init({
          wasmBase: "/mediapipe/wasm",
          modelBase: "/mediapipe/models",
          delegate,
          poseEvery,
          faceEvery,
        }),
      );
      runningRef.current = true;
      setStatus("running");
      setMessage("");

      while (runningRef.current) {
        const started = performance.now();
        const result = await pipeline.process(video, started);
        const finished = performance.now();
        s.frames += 1;
        s.done.push(finished);
        if (s.done.length > WINDOW) s.done.shift();
        s.endToEnd.push(finished - started);
        s.hand.push(result.timingsMs.hand);
        s.pose.push(result.timingsMs.pose);
        s.face.push(result.timingsMs.face);
        s.total.push(result.timingsMs.total);
        s.present = result.present;
      }
    } catch (error) {
      // A failure after stop() (pipeline disposed mid-frame) is expected, not an error.
      if (runningRef.current || pipelineRef.current) {
        setMessage(String(error));
        setStatus("error");
      }
      runningRef.current = false;
    }
  };

  const downloadReport = () => {
    const report = {
      when: new Date().toISOString(),
      userAgent: navigator.userAgent,
      logicalCores: navigator.hardwareConcurrency,
      options: { delegate, usedDelegate, poseEvery, faceEvery },
      targetFps: TARGET_FPS,
      ...snapshot(),
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "landmark-benchmark.json";
    link.click();
    URL.revokeObjectURL(url);
  };

  const verdict =
    stats && stats.frames >= MIN_FRAMES_FOR_VERDICT && Number.isFinite(stats.fps)
      ? stats.fps >= TARGET_FPS
        ? "PASS"
        : "FAIL"
      : "…";

  return (
    <main
      style={{
        fontFamily: "system-ui, sans-serif",
        maxWidth: 720,
        margin: "1rem auto",
        padding: "0 1rem",
      }}
    >
      <h1>Landmark benchmark</h1>
      <p>
        Camera frames are processed in this browser tab only and are never uploaded. Keep both hands
        and your face in view while measuring.
      </p>

      <video
        ref={videoRef}
        muted
        playsInline
        style={{ width: "100%", maxWidth: 640, background: "#111", transform: "scaleX(-1)" }}
      />

      <fieldset disabled={status === "loading" || status === "running"}>
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
        {status === "running" ? (
          <button onClick={stop}>Stop</button>
        ) : (
          <button onClick={() => void start()} disabled={status === "loading"}>
            Start camera and benchmark
          </button>
        )}{" "}
        <button onClick={downloadReport} disabled={!stats}>
          Download report (JSON)
        </button>
      </p>

      <p role="status" aria-live="polite">
        {message}
      </p>

      {stats && (
        <section aria-label="Results">
          <h2>
            {fmt(stats.fps)} fps — {verdict} (target ≥ {TARGET_FPS}, after {MIN_FRAMES_FOR_VERDICT}{" "}
            frames)
          </h2>
          <p>
            Delegate in use: <strong>{usedDelegate ?? "–"}</strong> · frames: {stats.frames}
          </p>
          <p>
            Detected now: left hand {stats.present?.leftHand ? "✔" : "✘"} · right hand{" "}
            {stats.present?.rightHand ? "✔" : "✘"} · pose {stats.present?.pose ? "✔" : "✘"} · face{" "}
            {stats.present?.face ? "✔" : "✘"}
          </p>
          <table>
            <thead>
              <tr>
                <th scope="col">Stage</th>
                <th scope="col">p50 ms</th>
                <th scope="col">p95 ms</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">End to end</th>
                <td>{fmt(stats.endToEndMs.p50)}</td>
                <td>{fmt(stats.endToEndMs.p95)}</td>
              </tr>
              {(["hand", "pose", "face", "total"] as const).map((stage) => (
                <tr key={stage}>
                  <th scope="row">Worker: {stage}</th>
                  <td>{fmt(stats.stageMs[stage].p50)}</td>
                  <td>{fmt(stats.stageMs[stage].p95)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </main>
  );
}
