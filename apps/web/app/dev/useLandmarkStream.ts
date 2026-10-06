"use client";

import {
  LandmarkPipeline,
  fpsFromTimestamps,
  type Delegate,
  type FrameResult,
  type Presence,
} from "@signlnk/landmarks";
import { useCallback, useEffect, useRef, useState } from "react";

export type StreamStatus = "idle" | "loading" | "running" | "error";

export interface StreamSettings {
  delegate: Delegate;
  poseEvery: number;
  faceEvery: number;
}

const FPS_WINDOW = 60;

/**
 * Camera + landmark worker for dev pages. Frames are processed in this tab only. `onFrame` is called
 * for every processed frame, and always sees the latest callback passed in.
 */
export function useLandmarkStream(onFrame: (result: FrameResult) => void) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const pipelineRef = useRef<LandmarkPipeline | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const runningRef = useRef(false);
  const onFrameRef = useRef(onFrame);
  const doneTimes = useRef<number[]>([]);
  const lastPresent = useRef<Presence | null>(null);

  const [status, setStatus] = useState<StreamStatus>("idle");
  const [message, setMessage] = useState("");
  const [usedDelegate, setUsedDelegate] = useState<Delegate | null>(null);
  const [fps, setFps] = useState(Number.NaN);
  const [present, setPresent] = useState<Presence | null>(null);

  useEffect(() => {
    onFrameRef.current = onFrame;
  });

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
    const timer = setInterval(() => {
      setFps(fpsFromTimestamps(doneTimes.current));
      setPresent(lastPresent.current);
    }, 500);
    return () => clearInterval(timer);
  }, [status]);

  const start = useCallback(async (settings: StreamSettings) => {
    doneTimes.current = [];
    lastPresent.current = null;
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
          ...settings,
        }),
      );
      runningRef.current = true;
      setStatus("running");
      setMessage("");

      while (runningRef.current) {
        const started = performance.now();
        const result = await pipeline.process(video, started);
        doneTimes.current.push(performance.now());
        if (doneTimes.current.length > FPS_WINDOW) doneTimes.current.shift();
        lastPresent.current = result.present;
        onFrameRef.current(result);
      }
    } catch (error) {
      // A failure after stop() (pipeline disposed mid-frame) is expected, not an error.
      if (runningRef.current || pipelineRef.current) {
        setMessage(String(error));
        setStatus("error");
      }
      runningRef.current = false;
    }
  }, []);

  return { videoRef, status, message, usedDelegate, fps, present, start, stop };
}
