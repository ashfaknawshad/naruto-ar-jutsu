/**
 * One Euro Filter (Casiez, Roussel, Vogel 2012) — low jitter when a signal
 * is nearly still, low lag when it moves fast. A plain moving average only
 * ever gives you one of those two properties; this is the standard fix and
 * it's what makes a tracked effect feel attached to the hand instead of
 * dragging behind it or trembling in place.
 */
export class OneEuroFilter {
  private minCutoff: number;
  private beta: number;
  private dCutoff: number;

  private lastValue: number | null = null;
  private lastDerivative = 0;
  private lastTimestamp: number | null = null;

  constructor(minCutoff = 1.0, beta = 0.0, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
  }

  private alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  /** @param timestampMs Monotonically increasing, e.g. performance.now(). */
  filter(value: number, timestampMs: number): number {
    if (this.lastTimestamp === null || this.lastValue === null) {
      this.lastTimestamp = timestampMs;
      this.lastValue = value;
      return value;
    }

    const dt = Math.max((timestampMs - this.lastTimestamp) / 1000, 1e-6);
    this.lastTimestamp = timestampMs;

    const derivative = (value - this.lastValue) / dt;
    const dAlpha = this.alpha(this.dCutoff, dt);
    const smoothedDerivative = dAlpha * derivative + (1 - dAlpha) * this.lastDerivative;
    this.lastDerivative = smoothedDerivative;

    // Cutoff rises with speed, so fast motion gets less smoothing (less
    // lag) while a still hand gets heavy smoothing (less jitter).
    const cutoff = this.minCutoff + this.beta * Math.abs(smoothedDerivative);
    const vAlpha = this.alpha(cutoff, dt);
    const smoothedValue = vAlpha * value + (1 - vAlpha) * this.lastValue;
    this.lastValue = smoothedValue;

    return smoothedValue;
  }

  reset(): void {
    this.lastValue = null;
    this.lastDerivative = 0;
    this.lastTimestamp = null;
  }
}

/** Three independent OneEuroFilters, one per axis. */
export class OneEuroVec3Filter {
  private x: OneEuroFilter;
  private y: OneEuroFilter;
  private z: OneEuroFilter;

  constructor(minCutoff = 1.0, beta = 0.0, dCutoff = 1.0) {
    this.x = new OneEuroFilter(minCutoff, beta, dCutoff);
    this.y = new OneEuroFilter(minCutoff, beta, dCutoff);
    this.z = new OneEuroFilter(minCutoff, beta, dCutoff);
  }

  filter(vx: number, vy: number, vz: number, timestampMs: number): [number, number, number] {
    return [this.x.filter(vx, timestampMs), this.y.filter(vy, timestampMs), this.z.filter(vz, timestampMs)];
  }

  reset(): void {
    this.x.reset();
    this.y.reset();
    this.z.reset();
  }
}
