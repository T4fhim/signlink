import { LANDMARK_COUNT, makeInput, normalizeFrame, resampleNearest } from "@signlnk/landmarks";
import type { RecognitionOutputV1 } from "@signlnk/schemas/recognition_output_v1";
import { Endpointer, type EndpointerConfig } from "./endpointer.ts";
import type { ModelInfo } from "./modelInfo.ts";

export interface DecoderConfig {
  model: ModelInfo;
  /** Sign language code of the model (config, never hard-coded), e.g. "ase". */
  language: string;
  /** SemVer of the model, copied into every caption. */
  modelVersion: string;
  /** CalVer of the lexicon bundle; "0000.0.0" until a bundle exists (Phase 1 step 7+). */
  lexiconVersion: string;
  /** Sign names (from the model's classes) that may be captioned; null = every sign. */
  vocabulary: readonly string[] | null;
  /** Minimum masked-softmax probability of the best sign. */
  minConfidence: number;
  /** The same gloss within this many ms of the previous one is not captioned again. */
  debounceMs: number;
  /** Candidates listed in `top_k`. */
  topK: number;
  endpointer?: Partial<EndpointerConfig>;
}

/** A finished sign, ready for the model: `input` is float32 [model.length * model.dim]. */
export interface PendingSign {
  input: Float32Array;
  tStart: number;
  tEnd: number;
  tDetected: number;
  frameCount: number;
  /** "max" = cut at maxSignMs while the hands still moved: not a measured sign end. */
  reason: "quiet" | "max";
}

/**
 * Live decoder core, free of the model runtime so it can run on any thread.
 *
 * `push` takes the landmark worker's frames (aspect-corrected, not yet normalized), finds the end
 * of each sign (Endpointer), resamples that segment to the model's window exactly as training did
 * and builds the model input. `interpret` turns the model's logits into a caption: softmax over the
 * vocabulary plus `<other>`, best sign above `minConfidence` and above `<other>`, debounced.
 */
export class Decoder {
  private readonly config: DecoderConfig;
  private readonly endpointer: Endpointer;
  private readonly signs: readonly number[];
  private last: { sign: number; tEnd: number } | null = null;

  constructor(config: DecoderConfig) {
    const { model } = config;
    if (model.nLandmarks !== LANDMARK_COUNT) {
      throw new Error(`model has ${model.nLandmarks} landmarks, the layout has ${LANDMARK_COUNT}`);
    }
    if (!(config.minConfidence >= 0 && config.minConfidence <= 1)) {
      throw new RangeError("minConfidence must be in [0, 1]");
    }
    if (!Number.isInteger(config.topK) || config.topK < 1) {
      throw new RangeError("topK must be a positive integer");
    }
    this.config = config;
    this.signs = resolveSigns(model, config.vocabulary);
    this.endpointer = new Endpointer(config.endpointer);
  }

  /** Feeds one landmark frame at `t` ms; returns a sign when one has just ended. */
  push(t: number, frame: Float32Array): PendingSign | null {
    const segment = this.endpointer.push(t, normalizeFrame(frame));
    if (!segment) return null;
    const { model } = this.config;
    const window = resampleNearest(segment.frames, model.nLandmarks * 3, model.length);
    return {
      input: makeInput(window, { useZ: model.useZ, nLandmarks: model.nLandmarks }),
      tStart: segment.tStart,
      tEnd: segment.tEnd,
      tDetected: segment.tDetected,
      frameCount: segment.frameCount,
      reason: segment.reason,
    };
  }

  /** Caption for a sign given the model's logits for it, or null when nothing should be shown. */
  interpret(sign: PendingSign, logits: Float32Array): RecognitionOutputV1 | null {
    const { model, language, topK } = this.config;
    if (logits.length !== model.classes.length) {
      throw new RangeError(`expected ${model.classes.length} logits, got ${logits.length}`);
    }
    const allowed = [...this.signs, model.otherIndex];
    const top = Math.max(...allowed.map((i) => logits[i] ?? Number.NEGATIVE_INFINITY));
    const weights = allowed.map((i) => Math.exp((logits[i] ?? Number.NEGATIVE_INFINITY) - top));
    const total = weights.reduce((a, b) => a + b, 0);
    const otherP = (weights.at(-1) ?? 0) / total;
    const ranked = this.signs
      .map((index, k) => ({ index, p: (weights[k] ?? 0) / total }))
      .sort((a, b) => b.p - a.p);
    const best = ranked[0];
    if (!best || otherP >= best.p || best.p < this.config.minConfidence) return null;

    const previous = this.last;
    this.last = { sign: best.index, tEnd: sign.tEnd };
    if (
      previous &&
      previous.sign === best.index &&
      sign.tStart - previous.tEnd < this.config.debounceMs
    ) {
      return null;
    }
    const glossId = (index: number): string => `${language}:${this.glossOf(index)}`;
    return {
      gloss: this.glossOf(best.index),
      gloss_id: glossId(best.index),
      confidence: best.p,
      t_start: Math.round(sign.tStart),
      t_end: Math.round(sign.tEnd),
      top_k: ranked.slice(0, topK).map((r) => ({ gloss_id: glossId(r.index), p: r.p })),
      model_version: this.config.modelVersion,
      lexicon_version: this.config.lexiconVersion,
    };
  }

  /** Forgets the open sign and the debounce state (camera restart, vocabulary change). */
  reset(): void {
    this.endpointer.reset();
    this.last = null;
  }

  /** ID-gloss of a class: the model's sign name, upper-cased. Temporary until the lexicon maps names. */
  private glossOf(index: number): string {
    return (this.config.model.classes[index] ?? "").toUpperCase();
  }
}

function resolveSigns(model: ModelInfo, vocabulary: readonly string[] | null): number[] {
  if (vocabulary === null) {
    return model.classes.map((_, i) => i).filter((i) => i !== model.otherIndex);
  }
  const seen = new Set<string>();
  return vocabulary.map((name) => {
    const index = model.classes.indexOf(name);
    if (index === -1) throw new Error(`vocabulary: unknown sign "${name}"`);
    if (index === model.otherIndex) throw new Error(`vocabulary: must not contain ${name}`);
    if (seen.has(name)) throw new Error(`vocabulary: repeated sign "${name}"`);
    seen.add(name);
    return index;
  });
}
