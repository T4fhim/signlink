import { inputDim } from "@signlnk/landmarks";

/** What the decoder needs from `model.json` (written by ml/signlnk_ml/export/onnx_export.py). */
export interface ModelInfo {
  /** Sign names in class order, then `<other>`. */
  classes: readonly string[];
  /** Index of `<other>` ("no sign"), the last class. */
  otherIndex: number;
  /** Frames per model window. */
  length: number;
  /** Values per frame row. */
  dim: number;
  useZ: boolean;
  nLandmarks: number;
}

const OTHER = "<other>";

const fail = (why: string): never => {
  throw new Error(`model.json: ${why}`);
};
const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);
const positiveInt = (value: unknown, name: string): number => {
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    return fail(`${name} must be a positive integer`);
  }
  return value;
};

/**
 * Validates a parsed `model.json` and returns what the decoder uses. It refuses what the TS input
 * builder (modelInput.ts) does not reproduce, so a mismatch fails loudly instead of mis-predicting.
 */
export function parseModelInfo(json: unknown): ModelInfo {
  if (!isRecord(json)) return fail("not an object");
  if (json["format"] !== "slk-recognizer-v1") {
    return fail(`unsupported format ${String(json["format"])}`);
  }
  const input = json["input"];
  if (!isRecord(input)) return fail("input is missing");
  if (input["name"] !== "windows") return fail(`unexpected input name ${String(input["name"])}`);
  if (input["layout"] !== "slk-landmarks-v1") {
    return fail(`unsupported layout ${String(input["layout"])}`);
  }
  if (input["velocity"] !== false) return fail("velocity models are not supported");
  const useZ = input["use_z"];
  if (typeof useZ !== "boolean") return fail("input.use_z must be a boolean");
  const nLandmarks = positiveInt(input["n_landmarks"], "input.n_landmarks");
  const length = positiveInt(input["length"], "input.length");
  const dim = positiveInt(input["dim"], "input.dim");
  if (dim !== inputDim(nLandmarks, useZ)) {
    return fail(`input.dim ${dim} does not match n_landmarks ${nLandmarks} with use_z ${useZ}`);
  }

  const classes = json["classes"];
  if (!Array.isArray(classes) || classes.length === 0) return fail("classes is empty");
  if (!classes.every((c): c is string => typeof c === "string")) {
    return fail("classes must be strings");
  }
  if (new Set(classes).size !== classes.length) return fail("classes has duplicates");
  const otherIndex = json["other_index"];
  if (typeof otherIndex !== "number" || classes[otherIndex] !== OTHER) {
    return fail(`other_index must point at ${OTHER}`);
  }
  return { classes, otherIndex, length, dim, useZ, nLandmarks };
}
