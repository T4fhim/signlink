import { RollingWindow } from "@signlnk/landmarks";
import type { Outcome } from "./decoder/index.ts";

/**
 * Measurements for the live decoder's Done-when (PLAN §7 Phase 1 step 6): whole-run mean fps and
 * the sign-end to caption latency, from samples the page feeds in. All timestamps are on one clock
 * (the main thread's performance.now()). Nothing here is stored or sent; the page offers the
 * report as a user-initiated download.
 */

/** PLAN §7 Phase 1 step 6. */
export const TARGET_FPS = 25;
/** The fps target is a whole-run mean over at least this long. */
export const MIN_RUN_MS = 60_000;
/** PLAN §2.2: p95 sign end to caption visible. */
export const TARGET_P95_LATENCY_MS = 500;
/** Below this many measured captions a nearest-rank p95 is just the maximum, so "met" needs more. */
export const MIN_LATENCY_SAMPLES = 20;

export interface FrameSample {
  /** When the frame's landmarks were ready, ms. */
  doneAt: number;
  /** Landmarking time for the frame, ms. */
  landmarkMs: number;
  leftHand: boolean;
  rightHand: boolean;
}

export interface SignSample {
  /** Last moving frame of the sign (the latency anchor), ms. */
  tEnd: number;
  /** Capture time of the frame that closed the sign, ms. */
  tDetected: number;
  /** When the main thread received the sign, ms. */
  receivedAt: number;
  /** Model run time, ms. */
  inferMs: number;
  /** When the caption was painted, ms; null when nothing was captioned. */
  paintedAt: number | null;
  /** "max": cut while the hands still moved, so `tEnd` is not a real sign end (not measured). */
  reason: "quiet" | "max";
  /** First moving frame of the sign, ms (with `tEnd` gives the segment length the model saw). */
  tStart?: number;
  /** Frames in the segment before it was resampled. */
  frameCount?: number;
  /** What the decoder did with the sign, and the candidates behind it. */
  outcome?: Outcome;
  top?: { gloss: string; p: number }[];
  otherP?: number;
}

/** One row of the per-sign log: what was signed (as the model saw it) and why it was or was not captioned. */
export interface SignLogEntry {
  tEnd: number;
  /** Segment length, ms; null when the sample carries no start time. */
  durationMs: number | null;
  /** Frames in the segment; null when unknown. */
  frames: number | null;
  reason: "quiet" | "max";
  outcome: Outcome | null;
  top: { gloss: string; p: number }[];
  otherP: number | null;
  /** Sign end to painted caption, ms; null when not captioned or cut while moving. */
  endToCaptionMs: number | null;
}

export interface Quantiles {
  p50: number;
  p95: number;
  max: number;
  count: number;
}

export interface RunReport {
  /** First to last frame, ms. */
  durationMs: number;
  frames: number;
  /** Whole-run mean frames per second. */
  fps: number;
  /** Longest wait between two frames, ms. */
  longestGapMs: number;
  /** Share of frames (0..1) with the hand(s) detected: the conditions of the run. */
  hands: { left: number; right: number; either: number; both: number };
  landmarkMs: Quantiles;
  /** `forced` signs were cut at maxSignMs while moving; they are left out of the latency numbers. */
  signs: { total: number; captioned: number; forced: number };
  /** Sign end to painted caption, captioned signs that ended at rest: the number the target is about. */
  signEndToCaptionMs: Quantiles;
  /** Sign end to the frame that closed it (the endpointer's wait), signs that ended at rest. */
  decoderWaitMs: Quantiles;
  /** Closing frame to the sign arriving on the main thread (landmarks, queueing, model). */
  pipelineMs: Quantiles;
  inferMs: Quantiles;
  /** Every sign in order: the evidence for tuning the endpointer, threshold and vocabulary. */
  signLog: SignLogEntry[];
  /** How many signs ended each way (signs without a verdict are not counted). */
  outcomes: Record<Outcome, number>;
  targets: {
    fps: { target: number; minDurationMs: number; met: boolean };
    p95LatencyMs: { target: number; minSamples: number; met: boolean };
  };
}

function quantiles(values: readonly number[]): Quantiles {
  const window = new RollingWindow(Number.MAX_SAFE_INTEGER);
  for (const value of values) window.push(value);
  return {
    p50: window.median,
    p95: window.p95,
    max: values.length === 0 ? Number.NaN : Math.max(...values),
    count: values.length,
  };
}

function logEntry(s: SignSample): SignLogEntry {
  return {
    tEnd: s.tEnd,
    durationMs: s.tStart === undefined ? null : s.tEnd - s.tStart,
    frames: s.frameCount ?? null,
    reason: s.reason,
    outcome: s.outcome ?? null,
    top: s.top ?? [],
    otherP: s.otherP ?? null,
    endToCaptionMs: s.reason === "quiet" && s.paintedAt !== null ? s.paintedAt - s.tEnd : null,
  };
}

function countOutcomes(signs: readonly SignSample[]): Record<Outcome, number> {
  const counts: Record<Outcome, number> = {
    captioned: 0,
    other: 0,
    "low-confidence": 0,
    repeat: 0,
  };
  for (const s of signs) if (s.outcome !== undefined) counts[s.outcome]++;
  return counts;
}

export class RunStats {
  private frames: FrameSample[] = [];
  private signs: SignSample[] = [];

  frame(sample: FrameSample): void {
    this.frames.push(sample);
  }

  sign(sample: SignSample): void {
    this.signs.push(sample);
  }

  reset(): void {
    this.frames = [];
    this.signs = [];
  }

  report(): RunReport {
    const frames = this.frames;
    const first = frames[0]?.doneAt;
    const last = frames[frames.length - 1]?.doneAt;
    const durationMs = first === undefined || last === undefined ? 0 : last - first;
    const fps = durationMs > 0 ? ((frames.length - 1) * 1000) / durationMs : Number.NaN;

    let longestGapMs = 0;
    for (let i = 1; i < frames.length; i++) {
      longestGapMs = Math.max(
        longestGapMs,
        (frames[i]?.doneAt ?? 0) - (frames[i - 1]?.doneAt ?? 0),
      );
    }
    const share = (count: number): number =>
      frames.length === 0 ? Number.NaN : count / frames.length;
    const hands = {
      left: share(frames.filter((f) => f.leftHand).length),
      right: share(frames.filter((f) => f.rightHand).length),
      either: share(frames.filter((f) => f.leftHand || f.rightHand).length),
      both: share(frames.filter((f) => f.leftHand && f.rightHand).length),
    };

    const atRest = this.signs.filter((s) => s.reason === "quiet");
    const painted = atRest.flatMap((s) => (s.paintedAt === null ? [] : [s.paintedAt - s.tEnd]));
    const signEndToCaptionMs = quantiles(painted);
    return {
      durationMs,
      frames: frames.length,
      fps,
      longestGapMs,
      hands,
      landmarkMs: quantiles(frames.map((f) => f.landmarkMs)),
      signs: {
        total: this.signs.length,
        captioned: this.signs.filter((s) => s.paintedAt !== null).length,
        forced: this.signs.length - atRest.length,
      },
      signEndToCaptionMs,
      decoderWaitMs: quantiles(atRest.map((s) => s.tDetected - s.tEnd)),
      pipelineMs: quantiles(this.signs.map((s) => s.receivedAt - s.tDetected)),
      inferMs: quantiles(this.signs.map((s) => s.inferMs)),
      signLog: this.signs.map(logEntry),
      outcomes: countOutcomes(this.signs),
      targets: {
        fps: {
          target: TARGET_FPS,
          minDurationMs: MIN_RUN_MS,
          met: fps >= TARGET_FPS && durationMs >= MIN_RUN_MS,
        },
        p95LatencyMs: {
          target: TARGET_P95_LATENCY_MS,
          minSamples: MIN_LATENCY_SAMPLES,
          met:
            signEndToCaptionMs.count >= MIN_LATENCY_SAMPLES &&
            signEndToCaptionMs.p95 < TARGET_P95_LATENCY_MS,
        },
      },
    };
  }
}
