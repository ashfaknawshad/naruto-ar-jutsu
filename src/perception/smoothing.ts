import type { Label } from "../data/seals";

const WINDOW_SIZE = 10;
const ENTER_CONFIDENCE = 0.85;
const HOLD_CONFIDENCE = 0.6;

export interface HeldSeal {
  label: Label;
  /** Average confidence of `label` across the current window. */
  confidence: number;
}

/**
 * Turns raw per-frame classifications into a stable "held seal" signal.
 * Never trust a single frame — vote over a sliding window, and use
 * hysteresis so a seal is harder to enter than to keep holding. Without
 * this, the model flickers between neighbouring guesses at pose boundaries
 * and every effect looks broken at the edges (see the design plan's
 * rationale in naruto-jutsu-ar-project-plan.md §4.2 step 4).
 */
export class SealSmoother {
  private window: { label: Label; confidence: number }[] = [];
  private held: HeldSeal | null = null;

  /** Feed one frame's classification in; returns the current held seal (or null if nothing is confidently held). */
  push(label: Label, confidence: number): HeldSeal | null {
    this.window.push({ label, confidence });
    if (this.window.length > WINDOW_SIZE) this.window.shift();

    const votes = new Map<Label, { count: number; confidenceSum: number }>();
    for (const frame of this.window) {
      const entry = votes.get(frame.label) ?? { count: 0, confidenceSum: 0 };
      entry.count++;
      entry.confidenceSum += frame.confidence;
      votes.set(frame.label, entry);
    }

    let candidateLabel: Label | null = null;
    let candidateVotes = { count: 0, confidenceSum: 0 };
    for (const [label, v] of votes) {
      if (v.count > candidateVotes.count) {
        candidateLabel = label;
        candidateVotes = v;
      }
    }
    if (!candidateLabel) return this.held;

    const candidateAvgConfidence = candidateVotes.confidenceSum / candidateVotes.count;
    const isMajority = candidateVotes.count > this.window.length / 2;

    if (this.held?.label === candidateLabel) {
      // Already holding this one — the easier bar keeps it from dropping
      // out from ordinary confidence jitter.
      this.held =
        isMajority && candidateAvgConfidence >= HOLD_CONFIDENCE
          ? { label: candidateLabel, confidence: candidateAvgConfidence }
          : null;
    } else if (isMajority && candidateAvgConfidence >= ENTER_CONFIDENCE) {
      // Switching to a different label needs the harder bar, so a single
      // noisy frame can't hijack an already-held pose.
      this.held = { label: candidateLabel, confidence: candidateAvgConfidence };
    }
    // Otherwise: not enough evidence either way — keep whatever was held
    // (including null) rather than guessing.

    return this.held;
  }

  reset(): void {
    this.window = [];
    this.held = null;
  }
}
