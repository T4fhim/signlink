/** Fixed-size window of recent samples with percentile queries. */
export class RollingWindow {
  private readonly values: number[] = [];

  constructor(private readonly capacity: number) {}

  push(value: number): void {
    this.values.push(value);
    if (this.values.length > this.capacity) this.values.shift();
  }

  clear(): void {
    this.values.length = 0;
  }

  get count(): number {
    return this.values.length;
  }

  /** Nearest-rank percentile, p in [0, 100]. NaN when empty. */
  percentile(p: number): number {
    if (this.values.length === 0) return Number.NaN;
    const sorted = [...this.values].sort((a, b) => a - b);
    const rank = Math.ceil((p / 100) * sorted.length);
    return sorted[Math.min(sorted.length, Math.max(1, rank)) - 1] ?? Number.NaN;
  }

  get median(): number {
    return this.percentile(50);
  }

  get p95(): number {
    return this.percentile(95);
  }
}

/** Frames per second over a window of frame-completion timestamps (ms). NaN with fewer than 2. */
export function fpsFromTimestamps(timestampsMs: readonly number[]): number {
  const first = timestampsMs[0];
  const last = timestampsMs[timestampsMs.length - 1];
  if (first === undefined || last === undefined || timestampsMs.length < 2 || last <= first) {
    return Number.NaN;
  }
  return ((timestampsMs.length - 1) * 1000) / (last - first);
}
