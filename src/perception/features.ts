import type { HandLandmarkerResult, NormalizedLandmark } from "@mediapipe/tasks-vision";

const NUM_LANDMARKS = 21;
const PER_HAND_FEATURES = NUM_LANDMARKS * 3; // x, y, z per landmark
const WRIST = 0;
const MIDDLE_MCP = 9;
const FINGERTIPS = [4, 8, 12, 16, 20]; // thumb, index, middle, ring, pinky

// A real hand's wrist-to-middle-MCP span, in MediaPipe's 0-1 normalized
// image coordinates, doesn't collapse this small even at booth distance —
// MediaPipe's own hand detector needs a reasonably sized crop to fire at
// all. A span below this is a degenerate detection (foreshortened, hand
// edge-on to the camera, or a bad frame), and dividing by it in
// normalizeHand blows the output up to huge, useless values instead of
// throwing — a real recording session hit exactly this (features with
// |value| > 300 in an otherwise ~[-3, 3] range). Treat it as no hand.
const MIN_HAND_SCALE = 0.03;

// Layout: [left hand 63][right hand 63][leftPresent][rightPresent]
// [wristDistance][relativeRotation][5x cross-hand fingertip distances]
export const FEATURE_LENGTH = PER_HAND_FEATURES * 2 + 2 + 1 + 1 + FINGERTIPS.length; // 135

const LEFT_OFFSET = 0;
const RIGHT_OFFSET = PER_HAND_FEATURES;
const PRESENCE_OFFSET = PER_HAND_FEATURES * 2;
const WRIST_DIST_OFFSET = PRESENCE_OFFSET + 2;
const RELATIVE_ROTATION_OFFSET = WRIST_DIST_OFFSET + 1;
const FINGERTIP_DIST_OFFSET = RELATIVE_ROTATION_OFFSET + 1;

interface NormalizedHand {
  /** 63 floats: wrist-relative, scale- and rotation-normalized x/y/z per landmark. */
  vector: Float32Array;
  /** Orientation (radians) of the wrist -> middle-MCP axis before normalization, for cross-hand relative rotation. */
  angle: number;
  /** wrist-to-middle-MCP pixel span, used to put cross-hand distances on a shared, scale-invariant footing. */
  scale: number;
}

/**
 * Puts one hand's 21 landmarks into a canonical frame: translated so the
 * wrist sits at the origin, scaled by the wrist-to-middle-MCP span so hand
 * size (and camera distance) don't matter, and rotated so that same span
 * always points "up" — so a tilted seal still reads as the same seal.
 */
function normalizeHand(landmarks: NormalizedLandmark[]): NormalizedHand {
  const wrist = landmarks[WRIST];
  const mcp = landmarks[MIDDLE_MCP];
  const dx = mcp.x - wrist.x;
  const dy = mcp.y - wrist.y;
  const scale = Math.max(Math.hypot(dx, dy), 1e-6);

  // Rotation that maps the unit wrist->MCP vector onto (0, -1) ("up" in
  // screen space, where y grows downward). See derivation: for unit vector
  // u = (dx, dy) / scale, cosT = -u.y and sinT = -u.x achieves this.
  const ux = dx / scale;
  const uy = dy / scale;
  const cosT = -uy;
  const sinT = -ux;

  const vector = new Float32Array(PER_HAND_FEATURES);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    const p = landmarks[i];
    const rx = (p.x - wrist.x) / scale;
    const ry = (p.y - wrist.y) / scale;
    const rz = (p.z - wrist.z) / scale;
    vector[i * 3 + 0] = cosT * rx - sinT * ry;
    vector[i * 3 + 1] = sinT * rx + cosT * ry;
    vector[i * 3 + 2] = rz;
  }

  return { vector, angle: Math.atan2(dy, dx), scale };
}

function wrapAngle(a: number): number {
  // Normalize to [-pi, pi].
  return Math.atan2(Math.sin(a), Math.cos(a));
}

/**
 * Builds the fixed-length feature vector the classifier trains and infers
 * on, from a raw MediaPipe hand-landmarker result. Always FEATURE_LENGTH
 * long regardless of how many hands are present — missing hands are zeros,
 * flagged by the presence bits, so downstream code never branches on shape.
 */
export function buildFeatureVector(result: HandLandmarkerResult): Float32Array {
  const vector = new Float32Array(FEATURE_LENGTH);

  let left: NormalizedLandmark[] | null = null;
  let right: NormalizedLandmark[] | null = null;
  for (let i = 0; i < result.landmarks.length; i++) {
    const label = result.handedness[i]?.[0]?.categoryName;
    if (label === "Left" && !left) left = result.landmarks[i];
    else if (label === "Right" && !right) right = result.landmarks[i];
  }

  let leftNorm: NormalizedHand | null = null;
  let rightNorm: NormalizedHand | null = null;

  if (left) {
    const norm = normalizeHand(left);
    if (norm.scale >= MIN_HAND_SCALE) {
      leftNorm = norm;
      vector.set(norm.vector, LEFT_OFFSET);
      vector[PRESENCE_OFFSET] = 1;
    } else {
      left = null; // degenerate detection — treat as if this hand weren't there
    }
  }
  if (right) {
    const norm = normalizeHand(right);
    if (norm.scale >= MIN_HAND_SCALE) {
      rightNorm = norm;
      vector.set(norm.vector, RIGHT_OFFSET);
      vector[PRESENCE_OFFSET + 1] = 1;
    } else {
      right = null;
    }
  }

  // Cross-hand features carry most of the signal for two-handed seals, but
  // only mean something when both hands are visible.
  if (left && right && leftNorm && rightNorm) {
    const avgScale = (leftNorm.scale + rightNorm.scale) / 2;

    const wristDx = left[WRIST].x - right[WRIST].x;
    const wristDy = left[WRIST].y - right[WRIST].y;
    vector[WRIST_DIST_OFFSET] = Math.hypot(wristDx, wristDy) / avgScale;

    vector[RELATIVE_ROTATION_OFFSET] = wrapAngle(leftNorm.angle - rightNorm.angle) / Math.PI;

    for (let i = 0; i < FINGERTIPS.length; i++) {
      const tip = FINGERTIPS[i];
      const dx = left[tip].x - right[tip].x;
      const dy = left[tip].y - right[tip].y;
      vector[FINGERTIP_DIST_OFFSET + i] = Math.hypot(dx, dy) / avgScale;
    }
  }

  return vector;
}

/**
 * The horizontal-mirror twin of a feature vector — what the same physical
 * pose would look like performed with the opposite handedness (or seen in a
 * mirror). This is what makes "someone recorded a seal mirror-flipped, on
 * purpose or by beginner mistake" a non-issue rather than noise: a mirrored
 * seal is a real, valid way a visitor's hands can look, so training on both
 * orientations (see train/train_mlp.py) makes the classifier invariant to
 * it instead of quietly overfitting to whichever chirality happened to get
 * recorded.
 *
 * Derivation: negating every landmark's raw x before normalizeHand() is
 * algebraically equivalent to negating just the x component of its already-
 * normalized output and leaving y/z untouched (the rotation-normalization
 * step's cos/sin terms work out so the two commute) — so this function can
 * operate directly on stored feature vectors without the original landmarks.
 * Left/right hand slots swap because a mirrored right hand looks like a left
 * hand. The two cross-hand distance features are magnitudes (hypot), which
 * don't change under a reflection, so they're copied as-is; relative
 * rotation is a signed angle and flips sign.
 */
export function mirrorFeatures(vec: ArrayLike<number>): Float32Array {
  const out = new Float32Array(FEATURE_LENGTH);
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
  for (let i = 0; i < FINGERTIPS.length; i++) {
    out[FINGERTIP_DIST_OFFSET + i] = vec[FINGERTIP_DIST_OFFSET + i];
  }
  return out;
}
