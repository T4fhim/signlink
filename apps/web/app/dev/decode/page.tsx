"use client";

import type { Delegate, FrameResult } from "@signlnk/landmarks";
import { RecognizerClient, type SignEvent } from "@signlnk/recognizer/client";
import { DEFAULT_ENDPOINTER } from "@signlnk/recognizer/decoder";
import { RunStats, type RunReport } from "@signlnk/recognizer/stats";
import type { RecognitionOutputV1 } from "@signlnk/schemas/recognition_output_v1";
import { useCallback, useEffect, useRef, useState } from "react";
import { DECODER_IDENTITY } from "../decoderIdentity";
import { useLandmarkStream } from "../useLandmarkStream";

const SHOWN_CAPTIONS = 8;

interface Shown {
  id: number;
  caption: RecognitionOutputV1;
}

const fmt = (value: number, digits = 1): string =>
  Number.isFinite(value) ? value.toFixed(digits) : "–";
const pct = (fraction: number): string =>
  Number.isFinite(fraction) ? `${Math.round(fraction * 100)}%` : "–";

async function fetchOk(url: string): Promise<Response> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response;
}

export default function DecodePage() {
  const statsRef = useRef(new RunStats());
  const clientRef = useRef<RecognizerClient | null>(null);
  const nextId = useRef(0);
  const unpainted = useRef(new Map<number, SignEvent>());
  const [captions, setCaptions] = useState<Shown[]>([]);
  const [report, setReport] = useState<RunReport | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [threads, setThreads] = useState(1);
  const [hasVocabulary, setHasVocabulary] = useState(false);

  const [delegate, setDelegate] = useState<Delegate>("GPU");
  const [poseEvery, setPoseEvery] = useState(2);
  const [faceEvery, setFaceEvery] = useState(2);
  const [vocabulary, setVocabulary] = useState<"all" | "draft">("all");
  const [minConfidence, setMinConfidence] = useState(0.5);
  const [debounceMs, setDebounceMs] = useState(600);
  const [startSpeed, setStartSpeed] = useState(DEFAULT_ENDPOINTER.startSpeed);
  const [stopSpeed, setStopSpeed] = useState(DEFAULT_ENDPOINTER.stopSpeed);
  const [stopMs, setStopMs] = useState(DEFAULT_ENDPOINTER.stopMs);

  useEffect(() => {
    void fetch("/model/vocabulary.json", { method: "HEAD" })
      .then((response) => setHasVocabulary(response.ok))
      .catch(() => setHasVocabulary(false));
  }, []);

  const onFrame = useCallback((result: FrameResult) => {
    clientRef.current?.pushFrame(result.t, result.frame);
    statsRef.current.frame({
      doneAt: performance.now(),
      landmarkMs: result.timingsMs.total,
      leftHand: result.present.leftHand,
      rightHand: result.present.rightHand,
    });
  }, []);

  const { videoRef, status, message, usedDelegate, fps, present, start, stop } =
    useLandmarkStream(onFrame);
  const running = status === "running";

  const onSign = useCallback((event: SignEvent) => {
    const { caption } = event;
    if (!caption) {
      statsRef.current.sign({ ...event, paintedAt: null });
      return;
    }
    const id = nextId.current++;
    unpainted.current.set(id, event);
    setCaptions((previous) => [{ id, caption }, ...previous].slice(0, SHOWN_CAPTIONS));
  }, []);

  // Paint time: two animation frames after React committed the caption.
  useEffect(() => {
    for (const [id, event] of unpainted.current) {
      unpainted.current.delete(id);
      requestAnimationFrame(() =>
        requestAnimationFrame(() =>
          statsRef.current.sign({ ...event, paintedAt: performance.now() }),
        ),
      );
    }
  }, [captions]);

  // The camera failed after the model loaded: do not keep the worker (and its memory) around.
  useEffect(() => {
    if (status !== "error") return;
    clientRef.current?.dispose();
    clientRef.current = null;
  }, [status]);

  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => setReport(statsRef.current.report()), 1000);
    return () => clearInterval(timer);
  }, [running]);

  useEffect(
    () => () => {
      clientRef.current?.dispose();
      clientRef.current = null;
    },
    [],
  );

  const begin = async () => {
    setError("");
    setLoading(true);
    try {
      const [modelJson, model, signs] = await Promise.all([
        fetchOk("/model/model.json").then((r) => r.json() as Promise<unknown>),
        fetchOk("/model/model.onnx").then(async (r) => new Uint8Array(await r.arrayBuffer())),
        vocabulary === "draft"
          ? fetchOk("/model/vocabulary.json").then((r) => r.json() as Promise<string[]>)
          : Promise.resolve(null),
      ]);
      clientRef.current?.dispose();
      statsRef.current.reset();
      setCaptions([]);
      setReport(null);
      const client = await RecognizerClient.create({
        model,
        modelJson,
        wasmBase: "/ort/",
        decoder: {
          ...DECODER_IDENTITY,
          vocabulary: signs,
          minConfidence,
          debounceMs,
          topK: 3,
          endpointer: { startSpeed, stopSpeed, stopMs },
        },
        onSign,
        onError: (e) => setError(e.message),
      });
      clientRef.current = client;
      setThreads(client.threads);
      void start({ delegate, poseEvery, faceEvery });
    } catch (e) {
      setError(
        `${e instanceof Error ? e.message : String(e)}. Stage a model first: ` +
          "node scripts/stage-model.mjs <run-dir> --vocab ml/vocab/draft-meeting-wishlist.yaml, then pnpm assets.",
      );
    } finally {
      setLoading(false);
    }
  };

  const end = () => {
    stop();
    clientRef.current?.dispose();
    clientRef.current = null;
    setReport(statsRef.current.report());
  };

  const downloadReport = () => {
    const body = {
      when: new Date().toISOString(),
      userAgent: navigator.userAgent,
      logicalCores: navigator.hardwareConcurrency,
      crossOriginIsolated: window.crossOriginIsolated,
      ortThreads: threads,
      options: {
        delegate,
        usedDelegate,
        poseEvery,
        faceEvery,
        vocabulary,
        minConfidence,
        debounceMs,
        endpointer: { ...DEFAULT_ENDPOINTER, startSpeed, stopSpeed, stopMs },
      },
      report: statsRef.current.report(),
    };
    const blob = new Blob([JSON.stringify(body, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "decode-benchmark.json";
    link.click();
    URL.revokeObjectURL(url);
  };

  const busy = loading || status === "loading" || running;
  const latency = report?.signEndToCaptionMs;

  return (
    <main
      style={{
        fontFamily: "system-ui, sans-serif",
        maxWidth: 760,
        margin: "1rem auto",
        padding: "0 1rem",
      }}
    >
      <h1>Live decoder</h1>
      <p>
        Camera video is processed in this browser tab only: landmarks are computed here, sent to a
        worker in this tab, and never uploaded. The report is a download you start. Machine
        recognition, not an interpreter.
      </p>

      <video
        ref={videoRef}
        muted
        playsInline
        style={{ width: "100%", maxWidth: 640, background: "#111", transform: "scaleX(-1)" }}
      />
      {(running || status === "loading") && (
        <p role="status" style={{ color: "#b00020" }}>
          {running ? "● Camera on" : "● Starting camera…"}
        </p>
      )}

      <section aria-label="Captions">
        <h2>Captions</h2>
        <ol
          role="log"
          aria-live="polite"
          style={{ listStyle: "none", padding: 0, minHeight: "6rem", color: "#111" }}
        >
          {captions.map(({ id, caption }, index) => (
            <li
              key={id}
              style={{
                fontSize: index === 0 ? "2.5rem" : "1.25rem",
                opacity: index === 0 ? 1 : 0.7,
              }}
            >
              <strong>{caption.gloss}</strong>{" "}
              <small>
                {pct(caption.confidence)} ·{" "}
                {caption.top_k.map((k) => k.gloss_id.split(":")[1]).join(", ")}
              </small>
            </li>
          ))}
        </ol>
      </section>

      <fieldset disabled={busy}>
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
        <br />
        <label>
          Vocabulary{" "}
          <select
            value={vocabulary}
            onChange={(e) => setVocabulary(e.target.value as "all" | "draft")}
          >
            <option value="all">all 250 signs</option>
            <option value="draft" disabled={!hasVocabulary}>
              draft meeting list (not approved)
            </option>
          </select>
        </label>{" "}
        <label>
          Min confidence{" "}
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={minConfidence}
            onChange={(e) => setMinConfidence(Number(e.target.value))}
            style={{ width: "5em" }}
          />
        </label>{" "}
        <label>
          Repeat gap ms{" "}
          <input
            type="number"
            min={0}
            step={100}
            value={debounceMs}
            onChange={(e) => setDebounceMs(Number(e.target.value))}
            style={{ width: "6em" }}
          />
        </label>
        <br />
        <label>
          Start speed{" "}
          <input
            type="number"
            min={0}
            step={0.05}
            value={startSpeed}
            onChange={(e) => setStartSpeed(Number(e.target.value))}
            style={{ width: "5em" }}
          />
        </label>{" "}
        <label>
          Stop speed{" "}
          <input
            type="number"
            min={0}
            step={0.05}
            value={stopSpeed}
            onChange={(e) => setStopSpeed(Number(e.target.value))}
            style={{ width: "5em" }}
          />
        </label>{" "}
        <label>
          Stop wait ms{" "}
          <input
            type="number"
            min={0}
            step={50}
            value={stopMs}
            onChange={(e) => setStopMs(Number(e.target.value))}
            style={{ width: "6em" }}
          />
        </label>
        <p>
          Speeds are shoulder widths per second (<code>[VERIFY]</code> defaults, tune on real
          signing). Stop wait is the decoder&apos;s share of the latency budget.
        </p>
      </fieldset>

      <p>
        {running ? (
          <button onClick={end}>Stop</button>
        ) : (
          <button onClick={() => void begin()} disabled={busy}>
            Start camera and decoder
          </button>
        )}{" "}
        <button onClick={downloadReport} disabled={!report || report.frames === 0}>
          Download report (JSON)
        </button>
        <br />
        <small>
          The report holds timings and settings, your browser version, core count and the time. It
          holds no video, landmarks or captions. The camera stays on until you press Stop.
        </small>
      </p>

      <p role="status" aria-live="polite">
        {loading && "Loading model…"}
        {message}
      </p>
      {error && (
        <p role="alert" style={{ color: "#b00020" }}>
          {error}
        </p>
      )}

      {report && (
        <section aria-label="Run">
          <h2>Run</h2>
          <p>
            <strong>{fmt(report.fps)} fps</strong> whole run over {fmt(report.durationMs / 1000, 0)}{" "}
            s (now {fmt(fps)}; target ≥ 25 over ≥ 60 s: {report.targets.fps.met ? "met" : "not yet"}
            ) · landmarks p50 {fmt(report.landmarkMs.p50)} / p95 {fmt(report.landmarkMs.p95)} ms ·
            delegate {usedDelegate ?? "–"} · ORT threads {threads}
            {typeof window !== "undefined" && window.crossOriginIsolated ? " (isolated)" : ""}
          </p>
          <p>
            Hands in view: left {pct(report.hands.left)} · right {pct(report.hands.right)} · both{" "}
            {pct(report.hands.both)} · now left {present?.leftHand ? "✔" : "✘"} right{" "}
            {present?.rightHand ? "✔" : "✘"}
          </p>
          <p>
            Signs {report.signs.total} ({report.signs.captioned} captioned, {report.signs.forced}{" "}
            cut while moving, left out of latency) · sign end → caption p50{" "}
            {fmt(latency?.p50 ?? Number.NaN, 0)} / p95 {fmt(latency?.p95 ?? Number.NaN, 0)} ms over{" "}
            {latency?.count ?? 0} (target &lt; 500 over ≥ {report.targets.p95LatencyMs.minSamples}:{" "}
            {report.targets.p95LatencyMs.met ? "met" : "not yet"}) · decoder wait p50{" "}
            {fmt(report.decoderWaitMs.p50, 0)} · pipeline p50 {fmt(report.pipelineMs.p50, 0)} ·
            model p50 {fmt(report.inferMs.p50, 0)} ms
          </p>
        </section>
      )}
    </main>
  );
}
