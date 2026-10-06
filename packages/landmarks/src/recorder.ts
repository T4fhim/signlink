import type { Presence } from "./frame.ts";
import type { FrameResult } from "./protocol.ts";

export interface RecordingSummary {
  frames: number;
  /** Time between the first and last frame, ms. */
  durationMs: number;
  fps: number;
  /** Fraction of recorded frames in which each part was detected. */
  presence: Record<keyof Presence, number>;
}

/** Collects landmark frames until a target duration has elapsed. Holds everything in memory. */
export class Recorder {
  private readonly frames: Float32Array[] = [];
  private readonly detected: Record<keyof Presence, number> = {
    leftHand: 0,
    rightHand: 0,
    pose: 0,
    face: 0,
  };
  private firstT: number | null = null;
  private lastT = 0;
  private complete = false;

  constructor(
    private readonly frameLength: number,
    private readonly targetMs: number,
  ) {}

  get isComplete(): boolean {
    return this.complete;
  }

  get count(): number {
    return this.frames.length;
  }

  /** Adds a frame; returns true once the target duration is reached. Later frames are ignored. */
  add(result: FrameResult): boolean {
    if (this.complete) return true;
    if (result.frame.length !== this.frameLength) {
      throw new RangeError(`expected ${this.frameLength} values, got ${result.frame.length}`);
    }
    this.firstT ??= result.t;
    this.lastT = result.t;
    this.frames.push(result.frame);
    for (const part of Object.keys(this.detected) as (keyof Presence)[]) {
      if (result.present[part]) this.detected[part] += 1;
    }
    this.complete = this.lastT - this.firstT >= this.targetMs;
    return this.complete;
  }

  summary(): RecordingSummary {
    const frames = this.frames.length;
    const durationMs = this.firstT === null ? 0 : this.lastT - this.firstT;
    const share = (n: number): number => (frames === 0 ? 0 : n / frames);
    return {
      frames,
      durationMs,
      fps: frames > 1 && durationMs > 0 ? ((frames - 1) * 1000) / durationMs : Number.NaN,
      presence: {
        leftHand: share(this.detected.leftHand),
        rightHand: share(this.detected.rightHand),
        pose: share(this.detected.pose),
        face: share(this.detected.face),
      },
    };
  }

  /** All frames concatenated row-major: float32 [frames * frameLength]. */
  toFloat32(): Float32Array {
    const out = new Float32Array(this.frames.length * this.frameLength);
    this.frames.forEach((frame, i) => out.set(frame, i * this.frameLength));
    return out;
  }
}
