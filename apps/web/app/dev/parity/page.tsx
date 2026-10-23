"use client";

import { RollingWindow } from "@signlnk/landmarks";
import { RecognizerClient } from "@signlnk/recognizer/client";
import { decodeParityFile, maxAbsDiff } from "@signlnk/recognizer/parity";
import { useState } from "react";
import { DECODER_IDENTITY } from "../decoderIdentity";

const SINGLE_RUNS = 30;

interface Result {
  batch: number;
  maxAbsLogits: number;
  maxAbsEmbedding: number;
  tolerance: number;
  pass: boolean;
  inferMs: number;
  singleWindowMs: { runs: number; p50: number; p95: number };
  ortThreads: number;
  crossOriginIsolated: boolean;
  /** Cross-origin requests the page itself made (the worker's own fetches show in DevTools). */
  foreignRequests: string[];
  userAgent: string;
}

async function fetchOk(url: string): Promise<Response> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response;
}

/**
 * Browser half of the ONNX parity check (PLAN Phase 1 step 5/6): runs the staged model's saved
 * windows through the inference worker (onnxruntime-web/wasm) and compares with PyTorch's logits.
 * Stage a model first: node scripts/stage-model.mjs <run-dir>.
 */
export default function ParityPage() {
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const [running, setRunning] = useState(false);

  const run = async () => {
    setError("");
    setResult(null);
    setRunning(true);
    try {
      const [modelJson, model, parityJson] = await Promise.all([
        fetchOk("/model/model.json").then((r) => r.json() as Promise<unknown>),
        fetchOk("/model/model.onnx").then(async (r) => new Uint8Array(await r.arrayBuffer())),
        fetchOk("/model/parity.json").then((r) => r.json() as Promise<unknown>),
      ]);
      const parity = decodeParityFile(parityJson);
      const client = await RecognizerClient.create({
        model,
        modelJson,
        wasmBase: "/ort/",
        decoder: {
          ...DECODER_IDENTITY,
          vocabulary: null,
          minConfidence: 0.5,
          debounceMs: 600,
          topK: 3,
        },
        onSign: () => undefined,
      });
      try {
        const batch = parity.windows.shape[0] ?? 0;
        const reply = await client.infer(parity.windows.data, batch);
        const maxAbsLogits = maxAbsDiff(reply.logits, parity.logits.data);
        const maxAbsEmbedding = maxAbsDiff(reply.embedding, parity.embedding.data);
        // One window per call is what the live decoder does (PLAN §2.2 budget: <= 40 ms).
        const one = parity.windows.data.slice(0, parity.windows.data.length / batch);
        await client.infer(one, 1); // warm-up
        const times = new RollingWindow(SINGLE_RUNS);
        for (let i = 0; i < SINGLE_RUNS; i += 1) times.push((await client.infer(one, 1)).inferMs);
        setResult({
          batch,
          maxAbsLogits,
          maxAbsEmbedding,
          tolerance: parity.tolerance,
          pass: maxAbsLogits <= parity.tolerance && maxAbsEmbedding <= parity.tolerance,
          inferMs: reply.inferMs,
          singleWindowMs: { runs: SINGLE_RUNS, p50: times.median, p95: times.p95 },
          ortThreads: client.threads,
          crossOriginIsolated: window.crossOriginIsolated,
          foreignRequests: performance
            .getEntriesByType("resource")
            .map((entry) => entry.name)
            .filter((url) => new URL(url).origin !== location.origin),
          userAgent: navigator.userAgent,
        });
      } finally {
        client.dispose();
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
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
      <h1>Browser parity check</h1>
      <p>
        Runs the staged model on its saved test windows in a Web Worker (onnxruntime-web, WASM) and
        compares the logits with PyTorch&apos;s. No camera is used and nothing leaves this tab.
      </p>
      <p>
        <button onClick={() => void run()} disabled={running}>
          Run parity check
        </button>
      </p>
      <p role="status" aria-live="polite">
        {running && "Running…"}
        {result && (result.pass ? "PASS" : "FAIL")}
      </p>
      {error && (
        <p role="alert" style={{ color: "#b00020" }}>
          {error}
        </p>
      )}
      {result && <pre data-testid="parity-result">{JSON.stringify(result, null, 2)}</pre>}
    </main>
  );
}
