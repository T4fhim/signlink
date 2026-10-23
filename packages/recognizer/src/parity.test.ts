import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { decodeParityFile, maxAbsDiff, PARITY_TOLERANCE } from "./parity.ts";
import { createRecognizer } from "./session.ts";

const fixture = (name: string): Buffer =>
  readFileSync(new URL(`../../../tests/fixtures/onnx/${name}`, import.meta.url));

describe("parity file", () => {
  it("decodes the committed tiny fixture written by Python", () => {
    const parity = decodeParityFile(JSON.parse(fixture("tiny.parity.json").toString("utf8")));
    expect(parity.windows.shape).toEqual([4, 64, 438]);
    expect(parity.logits.shape).toEqual([4, 11]);
    expect(parity.embedding.shape).toEqual([4, 8]);
    expect(parity.windows.data.length).toBe(4 * 64 * 438);
    expect(parity.tolerance).toBe(PARITY_TOLERANCE);
    expect(PARITY_TOLERANCE).toBe(1e-4); // PLAN §7 Phase 1 step 5
  });

  it("rejects a file whose data does not match its shape", () => {
    const entry = {
      shape: [2, 2],
      b64: Buffer.from(new Float32Array(3).buffer).toString("base64"),
    };
    expect(() =>
      decodeParityFile({
        format: "slk-parity-v1",
        tolerance: 1e-4,
        windows: entry,
        logits: entry,
        embedding: entry,
      }),
    ).toThrow(/shape/);
  });

  it("measures the largest absolute difference", () => {
    expect(maxAbsDiff(new Float32Array([1, 2, 3]), new Float32Array([1, 2.5, 2]))).toBeCloseTo(1);
    expect(() => maxAbsDiff(new Float32Array(2), new Float32Array(3))).toThrow(/length/);
  });
});

describe("onnxruntime-web vs PyTorch (tiny fixture)", () => {
  it("matches PyTorch's logits and embeddings within 1e-4", async () => {
    const parity = decodeParityFile(JSON.parse(fixture("tiny.parity.json").toString("utf8")));
    const [batch, length, dim] = parity.windows.shape as [number, number, number];
    const recognizer = await createRecognizer(new Uint8Array(fixture("tiny.onnx")), {
      length,
      dim,
    });
    const out = await recognizer.run(parity.windows.data, batch);
    expect(maxAbsDiff(out.logits, parity.logits.data)).toBeLessThanOrEqual(PARITY_TOLERANCE);
    expect(maxAbsDiff(out.embedding, parity.embedding.data)).toBeLessThanOrEqual(PARITY_TOLERANCE);
    await recognizer.dispose();
  });

  it("gives the same answer one window at a time, and stays finite on an empty window", async () => {
    const parity = decodeParityFile(JSON.parse(fixture("tiny.parity.json").toString("utf8")));
    const [batch, length, dim] = parity.windows.shape as [number, number, number];
    const recognizer = await createRecognizer(new Uint8Array(fixture("tiny.onnx")), {
      length,
      dim,
    });
    const classes = parity.logits.shape[1] as number;
    const each = new Float32Array(batch * classes);
    for (let i = 0; i < batch; i += 1) {
      const window = parity.windows.data.subarray(i * length * dim, (i + 1) * length * dim);
      each.set((await recognizer.run(window, 1)).logits, i * classes);
    }
    expect(maxAbsDiff(each, parity.logits.data)).toBeLessThanOrEqual(PARITY_TOLERANCE);
    expect(each.every(Number.isFinite)).toBe(true); // the last fixture window has no landmarks
    await recognizer.dispose();
  });

  it("refuses an input of the wrong size", async () => {
    const recognizer = await createRecognizer(new Uint8Array(fixture("tiny.onnx")), {
      length: 64,
      dim: 438,
    });
    await expect(recognizer.run(new Float32Array(10), 1)).rejects.toThrow(/size/);
    await recognizer.dispose();
  });
});

// The real trained model lives outside the repo; this test skips without SIGNLNK_DATA_DIR:
//   uv run python -m signlnk_ml.export.onnx_export --run-dir <run> \
//     --out-dir $SIGNLNK_DATA_DIR/models/baseline-seed1
const modelDir = process.env["SIGNLNK_DATA_DIR"]
  ? join(process.env["SIGNLNK_DATA_DIR"], "models", "baseline-seed1")
  : undefined;
const hasModel = modelDir !== undefined && existsSync(join(modelDir, "model.onnx"));

describe.skipIf(!hasModel)("onnxruntime-web vs PyTorch (trained baseline model)", () => {
  it("matches PyTorch's logits and embeddings within 1e-4", async () => {
    const dir = modelDir as string;
    const meta = JSON.parse(readFileSync(join(dir, "model.json"), "utf8")) as {
      classes: string[];
      input: { length: number; dim: number };
    };
    const parity = decodeParityFile(JSON.parse(readFileSync(join(dir, "parity.json"), "utf8")));
    const recognizer = await createRecognizer(
      new Uint8Array(readFileSync(join(dir, "model.onnx"))),
      meta.input,
    );
    const batch = parity.windows.shape[0] as number;
    const out = await recognizer.run(parity.windows.data, batch);
    expect(parity.logits.shape[1]).toBe(meta.classes.length);
    expect(maxAbsDiff(out.logits, parity.logits.data)).toBeLessThanOrEqual(PARITY_TOLERANCE);
    expect(maxAbsDiff(out.embedding, parity.embedding.data)).toBeLessThanOrEqual(PARITY_TOLERANCE);
    await recognizer.dispose();
  });
});
