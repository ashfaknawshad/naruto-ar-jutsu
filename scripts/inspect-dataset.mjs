// Merges every datasets/raw/*.json, reports coverage per label/person, and
// flags samples that look like genuine mistakes (empty detections, or
// far outliers within their class) — as opposed to a hand-sign recorded
// mirror-flipped, which is treated as valid data, not noise (see
// mirrorFeatures() in src/perception/features.ts for why that's sound).
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const rawDir = join(root, "datasets/raw");

const NUM_LANDMARKS = 21;
const PER_HAND_FEATURES = NUM_LANDMARKS * 3;
const FEATURE_LENGTH = PER_HAND_FEATURES * 2 + 2 + 1 + 1 + 5;
const LEFT_OFFSET = 0;
const RIGHT_OFFSET = PER_HAND_FEATURES;
const PRESENCE_OFFSET = PER_HAND_FEATURES * 2;
const WRIST_DIST_OFFSET = PRESENCE_OFFSET + 2;
const RELATIVE_ROTATION_OFFSET = WRIST_DIST_OFFSET + 1;
const FINGERTIP_DIST_OFFSET = RELATIVE_ROTATION_OFFSET + 1;

// Mirrors src/perception/features.ts::mirrorFeatures — kept duplicated here
// rather than imported so this script has zero build step.
function mirrorFeatures(vec) {
  const out = new Float64Array(FEATURE_LENGTH);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    out[LEFT_OFFSET + i * 3 + 0] = -vec[RIGHT_OFFSET + i * 3 + 0];
    out[LEFT_OFFSET + i * 3 + 1] = vec[RIGHT_OFFSET + i * 3 + 1];
    out[LEFT_OFFSET + i * 3 + 2] = vec[RIGHT_OFFSET + i * 3 + 2];
    out[RIGHT_OFFSET + i * 3 + 0] = -vec[LEFT_OFFSET + i * 3 + 0];
    out[RIGHT_OFFSET + i * 3 + 1] = vec[LEFT_OFFSET + i * 3 + 1];
    out[RIGHT_OFFSET + i * 3 + 2] = vec[LEFT_OFFSET + i * 3 + 2];
  }
  out[PRESENCE_OFFSET] = vec[PRESENCE_OFFSET + 1];
  out[PRESENCE_OFFSET + 1] = vec[PRESENCE_OFFSET];
  out[WRIST_DIST_OFFSET] = vec[WRIST_DIST_OFFSET];
  out[RELATIVE_ROTATION_OFFSET] = -vec[RELATIVE_ROTATION_OFFSET];
  for (let i = 0; i < 5; i++) out[FINGERTIP_DIST_OFFSET + i] = vec[FINGERTIP_DIST_OFFSET + i];
  return out;
}

function dist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

function mean(vectors) {
  const out = new Float64Array(FEATURE_LENGTH);
  for (const v of vectors) for (let i = 0; i < FEATURE_LENGTH; i++) out[i] += v[i] / vectors.length;
  return out;
}

// --- Load ------------------------------------------------------------------
const files = readdirSync(rawDir).filter((f) => f.endsWith(".json"));
if (files.length === 0) {
  console.log("no dataset files in datasets/raw/");
  process.exit(0);
}

const samples = []; // {file, index, label, person, features}
for (const file of files) {
  const parsed = JSON.parse(readFileSync(join(rawDir, file), "utf8"));
  if (parsed.featureLength !== FEATURE_LENGTH) {
    console.log(`skipping ${file}: featureLength ${parsed.featureLength} != ${FEATURE_LENGTH}`);
    continue;
  }
  parsed.samples.forEach((s, index) => {
    samples.push({ file, index, label: s.label, person: s.person, features: s.features });
  });
}

console.log(`loaded ${samples.length} samples from ${files.length} files\n`);

// --- Coverage: label x person ------------------------------------------------
const labels = [...new Set(samples.map((s) => s.label))].sort();
const people = [...new Set(samples.map((s) => s.person))].sort();

const grid = new Map(); // `${label}|${person}` -> count
for (const s of samples) {
  const key = `${s.label}|${s.person}`;
  grid.set(key, (grid.get(key) ?? 0) + 1);
}

console.log("coverage (samples per label per person):");
const nameWidth = Math.max(...labels.map((l) => l.length), 10) + 2;
console.log("".padEnd(nameWidth) + people.map((p) => p.padStart(9)).join("") + "   total");
for (const label of labels) {
  let total = 0;
  const row = people.map((p) => {
    const n = grid.get(`${label}|${p}`) ?? 0;
    total += n;
    return String(n || "-").padStart(9);
  });
  console.log(label.padEnd(nameWidth) + row.join("") + String(total).padStart(9));
}
console.log();

// --- Obvious defects: no hand detected at all -------------------------------
const noHands = samples.filter((s) => s.features[PRESENCE_OFFSET] === 0 && s.features[PRESENCE_OFFSET + 1] === 0);
if (noHands.length > 0) {
  console.log(`${noHands.length} samples have NO hand detected at all (camera missed the frame entirely):`);
  for (const s of noHands.slice(0, 20)) console.log(`  ${s.file} [${s.index}] ${s.person}/${s.label}`);
  if (noHands.length > 20) console.log(`  ... and ${noHands.length - 20} more`);
  console.log();
}

// --- Statistical outliers, mirror-aware -------------------------------------
// A sample recorded mirror-flipped isn't noise (see mirrorFeatures doc) —
// so for each sample we score it by whichever orientation (as-recorded, or
// mirrored) sits closer to its class's centroid, instead of penalizing every
// mirrored recording as if it were a mistake.
console.log("checking for likely mislabeled / noisy samples per class...\n");

for (const label of labels) {
  const inClass = samples.filter((s) => s.label === label && !(s.features[PRESENCE_OFFSET] === 0 && s.features[PRESENCE_OFFSET + 1] === 0));
  if (inClass.length < 8) continue; // too few to say anything statistically meaningful

  const pool = inClass.flatMap((s) => [s.features, Array.from(mirrorFeatures(s.features))]);
  const centroid = mean(pool);
  const poolDistances = pool.map((v) => dist(v, centroid));
  const avg = poolDistances.reduce((a, b) => a + b, 0) / poolDistances.length;
  const variance = poolDistances.reduce((a, b) => a + (b - avg) ** 2, 0) / poolDistances.length;
  const stdev = Math.sqrt(variance);
  const threshold = avg + 2.2 * stdev;

  const scored = inClass.map((s) => {
    const dOriginal = dist(s.features, centroid);
    const dMirrored = dist(mirrorFeatures(s.features), centroid);
    return { ...s, bestDist: Math.min(dOriginal, dMirrored), mirroredWasBetter: dMirrored < dOriginal };
  });

  const flagged = scored.filter((s) => s.bestDist > threshold).sort((a, b) => b.bestDist - a.bestDist);
  if (flagged.length > 0) {
    console.log(`"${label}" — ${flagged.length}/${inClass.length} samples far from the class centroid (threshold ${threshold.toFixed(2)}):`);
    for (const s of flagged.slice(0, 8)) {
      console.log(
        `  ${s.file} [${s.index}] ${s.person} — dist ${s.bestDist.toFixed(2)}${s.mirroredWasBetter ? " (closer if mirrored — likely just a flipped recording, probably fine)" : ""}`,
      );
    }
    if (flagged.length > 8) console.log(`  ... and ${flagged.length - 8} more`);
    console.log();
  }
}
