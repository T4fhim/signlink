import { LAYOUT } from "@signlnk/landmarks";
import type { LandmarkLayoutV1 } from "@signlnk/schemas/landmark_layout_v1";

/**
 * Motion-energy endpointing: finds where one sign starts and ends in a stream of normalized frames
 * (float32 [N * 3], shoulder-normalized, NaN = missing).
 *
 * Hand speed (mean x, y displacement of the hand landmarks per second, in shoulder widths) is
 * smoothed; it opens a segment after `startMs` above `startSpeed`, and closes it once it has been
 * below `stopSpeed` (or the hands were gone) for `stopMs`. The sign's end is the last frame that
 * was still moving, so the decoder's share of the latency budget is `stopMs`. Training windows are
 * whole clips resampled to 64 frames, so a segment keeps `prerollMs` before and `tailMs` after the
 * sign, as a clip would.
 */
export interface EndpointerConfig {
  /** Smoothed speed that counts as signing, shoulder widths per second. */
  startSpeed: number;
  /** Smoothed speed below which the hands count as at rest. */
  stopSpeed: number;
  /** Time above startSpeed before a segment opens. */
  startMs: number;
  /** Time at rest (or hands absent) before the segment closes. This is the decoder latency. */
  stopMs: number;
  /** Frames kept from before the first moving frame. */
  prerollMs: number;
  /** Frames kept after the last moving frame. */
  tailMs: number;
  /** Segments whose moving part is shorter than this are dropped as jitter. */
  minSignMs: number;
  /** A segment is closed at this length even if the hands keep moving. */
  maxSignMs: number;
  /** Weight of the newest speed sample in the exponential average, (0, 1]; 1 = no smoothing. */
  smoothing: number;
}

/** `[VERIFY]` Starting values, not tuned: calibrate on real signing with /dev/decode. */
export const DEFAULT_ENDPOINTER: EndpointerConfig = {
  startSpeed: 0.6,
  stopSpeed: 0.3,
  startMs: 90,
  stopMs: 200,
  prerollMs: 150,
  tailMs: 100,
  minSignMs: 250,
  maxSignMs: 4000,
  smoothing: 0.5,
};

export interface Segment {
  /** Normalized frames, float32 [frameCount * N * 3]. */
  frames: Float32Array;
  frameCount: number;
  /** First moving frame, ms on the caller's clock. */
  tStart: number;
  /** Last moving frame, ms: the anchor of the sign-end to caption latency. */
  tEnd: number;
  /** When the segment closed, ms. */
  tDetected: number;
  reason: "quiet" | "max";
}

/** Output-layout indices of the hand landmarks (both hands come first in slk-landmarks-v1). */
export function handIndices(layout: LandmarkLayoutV1 = LAYOUT): number[] {
  const out: number[] = [];
  let offset = 0;
  for (const group of layout.groups) {
    if (group.name.startsWith("hand_")) {
      for (let i = 0; i < group.indices.length; i++) out.push(offset + i);
    }
    offset += group.indices.length;
  }
  return out;
}

/**
 * Mean displacement per second of the landmarks present in both frames, in the frames' units.
 * 0 when no landmark is shared or time did not advance (hands appearing is not movement).
 */
export function handSpeed(
  previous: Float32Array,
  current: Float32Array,
  dtMs: number,
  indices: readonly number[],
): number {
  if (!(dtMs > 0)) return 0;
  let sum = 0;
  let count = 0;
  for (const i of indices) {
    const dx = (current[i * 3] ?? Number.NaN) - (previous[i * 3] ?? Number.NaN);
    const dy = (current[i * 3 + 1] ?? Number.NaN) - (previous[i * 3 + 1] ?? Number.NaN);
    if (Number.isFinite(dx) && Number.isFinite(dy)) {
      sum += Math.hypot(dx, dy);
      count++;
    }
  }
  return count === 0 ? 0 : sum / count / (dtMs / 1000);
}

interface Sample {
  t: number;
  frame: Float32Array;
}
interface Open {
  tStart: number;
  lastActive: number;
}

export class Endpointer {
  private readonly config: EndpointerConfig;
  private readonly hands: readonly number[];
  private previous: Sample | null = null;
  private speed: number | null = null;
  private buffer: Sample[] = [];
  private aboveSince: number | null = null;
  private open: Open | null = null;

  constructor(config: Partial<EndpointerConfig> = {}, hands: readonly number[] = handIndices()) {
    this.config = { ...DEFAULT_ENDPOINTER, ...config };
    this.hands = hands;
  }

  /** Feeds one normalized frame; returns a segment when a sign has just ended. */
  push(t: number, frame: Float32Array): Segment | null {
    const raw = this.previous
      ? handSpeed(this.previous.frame, frame, t - this.previous.t, this.hands)
      : 0;
    this.speed =
      this.speed === null ? raw : this.speed + this.config.smoothing * (raw - this.speed);
    this.previous = { t, frame };
    this.buffer.push({ t, frame });
    return this.open ? this.running(t, this.open, this.speed) : this.idle(t, this.speed);
  }

  /** Forgets any open segment and the previous frame (e.g. after the camera restarts). */
  reset(): void {
    this.previous = null;
    this.speed = null;
    this.buffer = [];
    this.aboveSince = null;
    this.open = null;
  }

  private idle(t: number, speed: number): null {
    const c = this.config;
    if (speed >= c.startSpeed) this.aboveSince ??= t;
    else this.aboveSince = null;
    const keepFrom = (this.aboveSince ?? t) - c.prerollMs;
    while (this.buffer.length > 1 && (this.buffer[0]?.t ?? t) < keepFrom) this.buffer.shift();
    if (this.aboveSince !== null && t - this.aboveSince >= c.startMs) {
      this.open = { tStart: this.aboveSince, lastActive: t };
    }
    return null;
  }

  private running(t: number, open: Open, speed: number): Segment | null {
    const c = this.config;
    if (speed >= c.stopSpeed) open.lastActive = t;
    if (t - open.lastActive >= c.stopMs) return this.close(open, "quiet", t);
    if (t - open.tStart >= c.maxSignMs) return this.close(open, "max", t);
    return null;
  }

  private close(open: Open, reason: Segment["reason"], tDetected: number): Segment | null {
    const c = this.config;
    const from = open.tStart - c.prerollMs;
    const to = open.lastActive + c.tailMs;
    const kept = this.buffer.filter((s) => s.t >= from && s.t <= to);
    this.buffer = [];
    this.aboveSince = null;
    this.open = null;
    const first = kept[0];
    if (!first || open.lastActive - open.tStart < c.minSignMs) return null;
    const frameLength = first.frame.length;
    const frames = new Float32Array(kept.length * frameLength);
    kept.forEach((s, i) => frames.set(s.frame, i * frameLength));
    return {
      frames,
      frameCount: kept.length,
      tStart: open.tStart,
      tEnd: open.lastActive,
      tDetected,
      reason,
    };
  }
}
