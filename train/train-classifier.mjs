// Trains the seal classifier from datasets/raw/*.json and writes
// public/models/weights.json for classifier.ts to load in the browser.
//
// No numpy/sklearn — the whole project is TS/JS and the dataset is already
// normalized landmark vectors (no image processing needed here), so a small
// hand-rolled MLP trained with plain-JS gradient descent is simpler than
// standing up a second language/toolchain for ~11k parameters.
//
// Two phases:
//  1. Person-grouped K-fold cross-validation, purely to REPORT a trustworthy
//     accuracy estimate. A single fixed 2-person test split swung 13+ points
//     depending on who got picked (one recorder's idiosyncratic technique
//     dominated the number) — with ~13 recorders total, that's expected:
//     any one small split is noisy. Folds are discarded after evaluation.
//  2. One final training run on ALL data (nothing held out) — that's the
//     model that actually ships. The CV estimate above is what tells you
//     how much to trust it; it doesn't produce the shipped weights itself.
//
// Run: node train/train-classifier.mjs
import { readFileSync, readdirSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { FEATURE_LENGTH, mirrorFeatures } from "../src/perception/features.ts";
import { ALL_LABELS } from "../src/data/seals.ts";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const rawDir = join(root, "datasets/raw");
const outPath = join(root, "public/models/weights.json");

const NUM_CLASSES = ALL_LABELS.length;
const HIDDEN1 = 64;
const HIDDEN2 = 32;
const EPOCHS = 120;
const BATCH_SIZE = 64;
const labelToIndex = new Map(ALL_LABELS.map((l, i) => [l, i]));

const K_FOLDS = 5;
// This session's dataset-recorder-name field, not a real person — full
// class coverage isn't there, so it's excluded from the fold grouping (kept
// in every fold's training data) rather than distorting per-class fold
// metrics with near-empty support.
const NON_PERSON = "transitions-and-none";

// Fixed base seed so runs are reproducible: weight init, shuffle order, and
// augmentation noise all used Math.random() before, so re-running on the
// *same* dataset could swing individual classes by 20+ points of F1 purely
// from training luck — indistinguishable from a real effect of new data
// unless the randomness is pinned down. mulberry32, tiny and dependency-free.
function mulberry32(seed) {
  let a = seed;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const BASE_SEED = 1337;

// --- Load + clean -----------------------------------------------------------
const files = readdirSync(rawDir).filter((f) => f.endsWith(".json"));
const all = [];
for (const file of files) {
  const parsed = JSON.parse(readFileSync(join(rawDir, file), "utf8"));
  if (parsed.featureLength !== FEATURE_LENGTH) continue;
  for (const s of parsed.samples) {
    // Presence flags live at indices 126/127 — drop frames where MediaPipe
    // found no hand at all (camera missed the frame), per inspect-dataset.mjs.
    if (s.features[126] === 0 && s.features[127] === 0) continue;
    // A handful of pre-fix recordings hit the degenerate-hand-scale bug that
    // MIN_HAND_SCALE in features.ts now guards against — those have feature
    // magnitudes in the hundreds instead of the normal ~[-3, 3] range.
    if (s.features.some((v) => Math.abs(v) > 10)) continue;
    if (!labelToIndex.has(s.label)) continue;
    all.push(s);
  }
}
console.log(`loaded ${all.length} usable samples from ${files.length} files`);

const people = [...new Set(all.map((s) => s.person))].filter((p) => p !== NON_PERSON);
console.log(`${people.length} recorders eligible for cross-validation folds: ${people.join(", ")}\n`);

// --- Shared building blocks (augmentation, model, training loop) -----------
// Mirror: a real, valid alternate-handedness version of the same pose (see
// mirrorFeatures' doc comment) — doubles the data for free and makes the
// classifier handedness-invariant. Jitter: small Gaussian noise on
// coordinates, standing in for landmark-detection noise. Small rotation:
// a further 2D rotation about the wrist origin, which composes cleanly
// with the normalization already applied (see notes in features.ts) and
// simulates natural tilt variation. Scale isn't augmented — the
// normalization pipeline already removes it, so resynthesizing it here
// would just be a no-op.
function jitter(rng, vec, sigma = 0.015) {
  const out = new Float64Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] + (rng() * 2 - 1) * sigma;
  return out;
}

function smallRotate(rng, vec, maxRadians = 0.12) {
  const out = Float64Array.from(vec);
  const angle = (rng() * 2 - 1) * maxRadians;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  for (const base of [0, 63]) {
    for (let i = 0; i < 21; i++) {
      const xi = base + i * 3;
      const yi = xi + 1;
      const x = out[xi];
      const y = out[yi];
      out[xi] = c * x - s * y;
      out[yi] = s * x + c * y;
    }
  }
  return out;
}

// none/transition have far fewer raw recordings than any seal (~207/217 vs
// 600-2000+) — a real session found this matters: ambiguous poses like a
// raised fist or a hand near the face were confidently misclassified as a
// seal instead of none, and person-grouped CV independently flagged
// ox<->transition as the single largest confusion in the whole matrix.
// Even split evenly, 4x augmentation leaves none/transition proportionally
// tiny. Boost their augmented-jitter reps so post-augmentation volume is
// roughly comparable to an average seal's, instead of oversampling by
// literal duplication (which would just be the same 207 poses repeated —
// each extra rep here draws fresh random jitter/rotation, so it's real
// synthetic variety, not copies).
const BOOST_LABELS = new Set(["none", "transition"]);
const NORMAL_JITTER_REPS = 1;
const BOOST_JITTER_REPS = 7; // -> 2 + 7*2 = 16 variants/sample, vs 2 + 1*2 = 4 normally

function augment(rng, samples) {
  const out = [];
  for (const s of samples) {
    const base = Float64Array.from(s.features);
    const mirrored = mirrorFeatures(base);
    out.push({ label: s.label, features: base });
    out.push({ label: s.label, features: mirrored });

    const reps = BOOST_LABELS.has(s.label) ? BOOST_JITTER_REPS : NORMAL_JITTER_REPS;
    for (let i = 0; i < reps; i++) {
      out.push({ label: s.label, features: jitter(rng, smallRotate(rng, base)) });
      out.push({ label: s.label, features: jitter(rng, smallRotate(rng, mirrored)) });
    }
  }
  return out;
}

function shuffle(rng, arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

function heInit(rng, rows, cols) {
  const scale = Math.sqrt(2 / rows);
  const m = new Float64Array(rows * cols);
  for (let i = 0; i < m.length; i++) m[i] = (rng() * 2 - 1) * scale;
  return m;
}

/** One independent model instance — each fold (and the final run) gets its own, so nothing leaks between them. */
function makeModel(rng) {
  const W1 = heInit(rng, FEATURE_LENGTH, HIDDEN1);
  const b1 = new Float64Array(HIDDEN1);
  const W2 = heInit(rng, HIDDEN1, HIDDEN2);
  const b2 = new Float64Array(HIDDEN2);
  const W3 = heInit(rng, HIDDEN2, NUM_CLASSES);
  const b3 = new Float64Array(NUM_CLASSES);
  const params = { W1, b1, W2, b2, W3, b3 };
  const m = {}, v = {};
  for (const k in params) {
    m[k] = new Float64Array(params[k].length);
    v[k] = new Float64Array(params[k].length);
  }
  const BETA1 = 0.9, BETA2 = 0.999, EPS = 1e-8, LR = 0.01, L2 = 1e-4;
  let adamT = 0;

  function forward(x) {
    const z1 = new Float64Array(HIDDEN1);
    for (let j = 0; j < HIDDEN1; j++) {
      let s = b1[j];
      for (let i = 0; i < FEATURE_LENGTH; i++) s += x[i] * W1[i * HIDDEN1 + j];
      z1[j] = Math.max(0, s);
    }
    const z2 = new Float64Array(HIDDEN2);
    for (let j = 0; j < HIDDEN2; j++) {
      let s = b2[j];
      for (let i = 0; i < HIDDEN1; i++) s += z1[i] * W2[i * HIDDEN2 + j];
      z2[j] = Math.max(0, s);
    }
    const logits = new Float64Array(NUM_CLASSES);
    for (let j = 0; j < NUM_CLASSES; j++) {
      let s = b3[j];
      for (let i = 0; i < HIDDEN2; i++) s += z2[i] * W3[i * NUM_CLASSES + j];
      logits[j] = s;
    }
    const max = Math.max(...logits);
    const exps = logits.map((v) => Math.exp(v - max));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((v) => v / sumExp);
    return { z1, z2, probs };
  }

  function predict(x) {
    const { probs } = forward(x);
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return best;
  }

  function accuracy(X, Y) {
    if (X.length === 0) return NaN;
    let correct = 0;
    for (let i = 0; i < X.length; i++) if (predict(X[i]) === Y[i]) correct++;
    return correct / X.length;
  }

  function adamStep(key, grad) {
    adamT++;
    const p = params[key];
    const mm = m[key];
    const vv = v[key];
    for (let i = 0; i < p.length; i++) {
      const g = grad[i] + L2 * p[i];
      mm[i] = BETA1 * mm[i] + (1 - BETA1) * g;
      vv[i] = BETA2 * vv[i] + (1 - BETA2) * g * g;
      const mHat = mm[i] / (1 - BETA1 ** adamT);
      const vHat = vv[i] / (1 - BETA2 ** adamT);
      p[i] -= (LR * mHat) / (Math.sqrt(vHat) + EPS);
    }
  }

  function trainBatch(batch) {
    const gW1 = new Float64Array(W1.length), gb1 = new Float64Array(b1.length);
    const gW2 = new Float64Array(W2.length), gb2 = new Float64Array(b2.length);
    const gW3 = new Float64Array(W3.length), gb3 = new Float64Array(b3.length);

    for (const { x, y } of batch) {
      const { z1, z2, probs } = forward(x);
      const dLogits = Float64Array.from(probs);
      dLogits[y] -= 1;

      for (let j = 0; j < NUM_CLASSES; j++) {
        gb3[j] += dLogits[j];
        for (let i = 0; i < HIDDEN2; i++) gW3[i * NUM_CLASSES + j] += z2[i] * dLogits[j];
      }
      const dz2 = new Float64Array(HIDDEN2);
      for (let i = 0; i < HIDDEN2; i++) {
        let s = 0;
        for (let j = 0; j < NUM_CLASSES; j++) s += W3[i * NUM_CLASSES + j] * dLogits[j];
        dz2[i] = z2[i] > 0 ? s : 0;
      }

      for (let j = 0; j < HIDDEN2; j++) {
        gb2[j] += dz2[j];
        for (let i = 0; i < HIDDEN1; i++) gW2[i * HIDDEN2 + j] += z1[i] * dz2[j];
      }
      const dz1 = new Float64Array(HIDDEN1);
      for (let i = 0; i < HIDDEN1; i++) {
        let s = 0;
        for (let j = 0; j < HIDDEN2; j++) s += W2[i * HIDDEN2 + j] * dz2[j];
        dz1[i] = z1[i] > 0 ? s : 0;
      }

      for (let j = 0; j < HIDDEN1; j++) {
        gb1[j] += dz1[j];
        for (let i = 0; i < FEATURE_LENGTH; i++) gW1[i * HIDDEN1 + j] += x[i] * dz1[j];
      }
    }

    const n = batch.length;
    for (const g of [gW1, gb1, gW2, gb2, gW3, gb3]) for (let i = 0; i < g.length; i++) g[i] /= n;

    adamStep("W1", gW1);
    adamStep("b1", gb1);
    adamStep("W2", gW2);
    adamStep("b2", gb2);
    adamStep("W3", gW3);
    adamStep("b3", gb3);
  }

  return { params, forward, predict, accuracy, trainBatch };
}

/**
 * Trains on `train`, early-stopping on a 10% validation slice, then
 * evaluates on `test` (may be empty — the final full-data run has no held-out
 * set). Returns the confusion matrix and the fold's trained weights.
 */
function runSplit(train, test, seed, { logEpochs = false } = {}) {
  const rng = mulberry32(seed);
  const trainAug = augment(rng, train);
  shuffle(rng, trainAug);

  const mean = new Float64Array(FEATURE_LENGTH);
  const std = new Float64Array(FEATURE_LENGTH);
  for (const s of train) for (let i = 0; i < FEATURE_LENGTH; i++) mean[i] += s.features[i] / train.length;
  for (const s of train) for (let i = 0; i < FEATURE_LENGTH; i++) std[i] += (s.features[i] - mean[i]) ** 2 / train.length;
  for (let i = 0; i < FEATURE_LENGTH; i++) std[i] = Math.sqrt(std[i]) || 1;
  const standardize = (vec) => {
    const out = new Float64Array(vec.length);
    for (let i = 0; i < vec.length; i++) out[i] = (vec[i] - mean[i]) / std[i];
    return out;
  };

  const valCount = Math.floor(trainAug.length * 0.1);
  const valSet = trainAug.slice(0, valCount);
  const fitSet = trainAug.slice(valCount);
  const X_fit = fitSet.map((s) => standardize(s.features));
  const Y_fit = fitSet.map((s) => labelToIndex.get(s.label));
  const X_val = valSet.map((s) => standardize(s.features));
  const Y_val = valSet.map((s) => labelToIndex.get(s.label));
  const X_test = test.map((s) => standardize(s.features));
  const Y_test = test.map((s) => labelToIndex.get(s.label));

  const model = makeModel(rng);
  let bestVal = -1;
  let bestParams = null;
  const indices = fitSet.map((_, i) => i);

  for (let epoch = 1; epoch <= EPOCHS; epoch++) {
    shuffle(rng, indices);
    for (let start = 0; start < indices.length; start += BATCH_SIZE) {
      const batch = indices.slice(start, start + BATCH_SIZE).map((i) => ({ x: X_fit[i], y: Y_fit[i] }));
      model.trainBatch(batch);
    }
    if (epoch % 5 === 0 || epoch === EPOCHS) {
      const valAcc = model.accuracy(X_val, Y_val);
      if (logEpochs) console.log(`  epoch ${epoch}: val accuracy ${(valAcc * 100).toFixed(1)}%`);
      if (valAcc > bestVal) {
        bestVal = valAcc;
        bestParams = Object.fromEntries(Object.entries(model.params).map(([k, v]) => [k, Float64Array.from(v)]));
      }
    }
  }
  Object.assign(model.params, bestParams);

  const confusion = Array.from({ length: NUM_CLASSES }, () => new Array(NUM_CLASSES).fill(0));
  for (let i = 0; i < X_test.length; i++) confusion[Y_test[i]][model.predict(X_test[i])]++;

  return { bestVal, confusion, testCount: X_test.length, weights: { mean, std, ...model.params } };
}

function summarize(confusion, label) {
  let correct = 0, total = 0;
  for (let c = 0; c < NUM_CLASSES; c++) for (let p = 0; p < NUM_CLASSES; p++) { total += confusion[c][p]; if (c === p) correct += confusion[c][p]; }
  console.log(`\n=== ${label} ===`);
  if (total === 0) {
    console.log("no test samples");
    return;
  }
  console.log(`overall accuracy: ${((correct / total) * 100).toFixed(1)}% (${total} samples)\n`);
  console.log("per-class precision / recall / F1:");
  for (let c = 0; c < NUM_CLASSES; c++) {
    const tp = confusion[c][c];
    const supportRow = confusion[c].reduce((a, b) => a + b, 0);
    if (supportRow === 0) continue;
    const predictedCol = confusion.reduce((a, row) => a + row[c], 0);
    const precision = predictedCol > 0 ? tp / predictedCol : NaN;
    const recall = tp / supportRow;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : NaN;
    console.log(
      `  ${ALL_LABELS[c].padEnd(11)} precision ${(precision * 100).toFixed(0).padStart(3)}%  recall ${(recall * 100).toFixed(0).padStart(3)}%  f1 ${(f1 * 100).toFixed(0).padStart(3)}%  (n=${supportRow})`,
    );
  }
  console.log("\nconfusion matrix (rows=true, cols=predicted):");
  const w = 6;
  console.log("".padEnd(12) + ALL_LABELS.map((l) => l.slice(0, 4).padStart(w)).join(""));
  for (let c = 0; c < NUM_CLASSES; c++) {
    if (confusion[c].reduce((a, b) => a + b, 0) === 0) continue;
    console.log(ALL_LABELS[c].padEnd(12) + confusion[c].map((n) => String(n).padStart(w)).join(""));
  }
  return correct / total;
}

// --- Phase 1: person-grouped K-fold cross-validation ------------------------
const groupRng = mulberry32(BASE_SEED);
const shuffledPeople = [...people];
shuffle(groupRng, shuffledPeople);
const folds = Array.from({ length: K_FOLDS }, () => []);
shuffledPeople.forEach((p, i) => folds[i % K_FOLDS].push(p));

console.log(`=== ${K_FOLDS}-fold cross-validation (grouped by recorder) ===`);
const pooledConfusion = Array.from({ length: NUM_CLASSES }, () => new Array(NUM_CLASSES).fill(0));
const foldAccuracies = [];

folds.forEach((foldPeople, i) => {
  if (foldPeople.length === 0) return;
  const testSet = new Set(foldPeople);
  const trainFold = all.filter((s) => !testSet.has(s.person));
  const testFold = all.filter((s) => testSet.has(s.person));
  console.log(`\nfold ${i + 1}/${K_FOLDS} — held out: ${foldPeople.join(", ")} (${testFold.length} samples)`);
  const { confusion, testCount, bestVal } = runSplit(trainFold, testFold, BASE_SEED + i + 1);
  console.log(`  fold val accuracy (same-person, sanity check): ${(bestVal * 100).toFixed(1)}%`);
  let correct = 0;
  for (let c = 0; c < NUM_CLASSES; c++) {
    for (let p = 0; p < NUM_CLASSES; p++) {
      pooledConfusion[c][p] += confusion[c][p];
      if (c === p) correct += confusion[c][p];
    }
  }
  const foldAcc = testCount > 0 ? correct / testCount : NaN;
  foldAccuracies.push(foldAcc);
  console.log(`  fold held-out accuracy: ${(foldAcc * 100).toFixed(1)}%`);
});

const meanAcc = foldAccuracies.reduce((a, b) => a + b, 0) / foldAccuracies.length;
const variance = foldAccuracies.reduce((a, b) => a + (b - meanAcc) ** 2, 0) / foldAccuracies.length;
console.log(`\ncross-validated accuracy: ${(meanAcc * 100).toFixed(1)}% ± ${(Math.sqrt(variance) * 100).toFixed(1)}% across ${foldAccuracies.length} folds`);
summarize(pooledConfusion, "pooled cross-validation results (all folds combined)");

// --- Phase 2: final model trained on everything ------------------------------
// This is what actually ships. The CV numbers above are the confidence
// estimate for it — no held-out set here because there's nothing left to
// hold out once we're using every sample.
console.log("\n=== training final model on all data ===");
const { weights } = runSplit(all, [], BASE_SEED, { logEpochs: true });

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(
  outPath,
  JSON.stringify({
    version: 1,
    labels: ALL_LABELS,
    featureLength: FEATURE_LENGTH,
    mean: Array.from(weights.mean),
    std: Array.from(weights.std),
    hidden1: HIDDEN1,
    hidden2: HIDDEN2,
    W1: Array.from(weights.W1),
    b1: Array.from(weights.b1),
    W2: Array.from(weights.W2),
    b2: Array.from(weights.b2),
    W3: Array.from(weights.W3),
    b3: Array.from(weights.b3),
  }),
);
console.log(`\nsaved final model -> ${outPath}`);
console.log(`(expected real-world accuracy: ~${(meanAcc * 100).toFixed(0)}% ± ${(Math.sqrt(variance) * 100).toFixed(0)}%, per the cross-validation above)`);
