import { ALL_LABELS, type Label } from "../data/seals";
import { FEATURE_LENGTH } from "./features";

interface Weights {
  version: 1;
  labels: readonly string[];
  featureLength: number;
  mean: number[];
  std: number[];
  hidden1: number;
  hidden2: number;
  W1: number[];
  b1: number[];
  W2: number[];
  b2: number[];
  W3: number[];
  b3: number[];
}

let weights: Weights | null = null;

export async function loadClassifier(): Promise<void> {
  const res = await fetch(`${import.meta.env.BASE_URL}models/weights.json`);
  if (!res.ok) throw new Error(`failed to load classifier weights: ${res.status}`);
  const w = (await res.json()) as Weights;
  if (w.featureLength !== FEATURE_LENGTH) {
    throw new Error(`weights.json feature length ${w.featureLength} != current FEATURE_LENGTH ${FEATURE_LENGTH}`);
  }
  weights = w;
}

export interface Classification {
  label: Label;
  confidence: number;
  /** Full softmax distribution, indexed the same as ALL_LABELS — smoothing.ts needs this, not just the top pick. */
  probs: Float64Array;
}

/**
 * Runs the same 135 -> hidden1 (relu) -> hidden2 (relu) -> 14 (softmax)
 * forward pass as train/train-classifier.mjs, using its trained weights.
 * Mirrors that script's math exactly — see it for the training side.
 */
export function classify(features: ArrayLike<number>): Classification {
  if (!weights) throw new Error("loadClassifier() must be awaited before classify()");
  const { mean, std, hidden1, hidden2, W1, b1, W2, b2, W3, b3 } = weights;
  const numClasses = ALL_LABELS.length;

  const x = new Float64Array(FEATURE_LENGTH);
  for (let i = 0; i < FEATURE_LENGTH; i++) x[i] = (features[i] - mean[i]) / std[i];

  const z1 = new Float64Array(hidden1);
  for (let j = 0; j < hidden1; j++) {
    let s = b1[j];
    for (let i = 0; i < FEATURE_LENGTH; i++) s += x[i] * W1[i * hidden1 + j];
    z1[j] = Math.max(0, s);
  }

  const z2 = new Float64Array(hidden2);
  for (let j = 0; j < hidden2; j++) {
    let s = b2[j];
    for (let i = 0; i < hidden1; i++) s += z1[i] * W2[i * hidden2 + j];
    z2[j] = Math.max(0, s);
  }

  const logits = new Float64Array(numClasses);
  for (let j = 0; j < numClasses; j++) {
    let s = b3[j];
    for (let i = 0; i < hidden2; i++) s += z2[i] * W3[i * numClasses + j];
    logits[j] = s;
  }

  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  const exps = new Float64Array(numClasses);
  let sumExp = 0;
  for (let j = 0; j < numClasses; j++) {
    exps[j] = Math.exp(logits[j] - max);
    sumExp += exps[j];
  }
  const probs = new Float64Array(numClasses);
  let best = 0;
  for (let j = 0; j < numClasses; j++) {
    probs[j] = exps[j] / sumExp;
    if (probs[j] > probs[best]) best = j;
  }

  return { label: ALL_LABELS[best], confidence: probs[best], probs };
}
