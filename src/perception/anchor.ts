import type { HandLandmarkerResult, Landmark, NormalizedLandmark } from "@mediapipe/tasks-vision";

const WRIST = 0;
const INDEX_MCP = 5;
const PINKY_MCP = 17;
// The rigid part of the hand — wrist plus the five MCP knuckles. Fingers
// articulate; this region doesn't, which is why MediaPipe's own palm
// detector is built around it too.
const PALM_LANDMARKS = [0, 5, 9, 13, 17];

// A typical adult palm span in metres. Only used to express `scale` in the
// same "fraction of frame" units the rest of the pipeline already assumed,
// so RADIUS_MULTIPLIER downstream keeps a sane magnitude — the actual
// rotation-invariance comes from the world-landmark ratio, not this number.
const REFERENCE_PALM_METRES = 0.09;

export interface HandAnchor {
  /** Palm centroid, in MediaPipe's normalized 0-1 image coordinates. */
  x: number;
  y: number;
  z: number;
  /** How large a REFERENCE_PALM_METRES object appears, as a fraction of the frame — i.e. purely a function of camera distance, not hand pose. See computeHandAnchor for how rotation is cancelled out. */
  scale: number;
  /** Unit palm-normal vector (image-space-relative, not camera-calibrated). */
  normal: { x: number; y: number; z: number };
}

export interface HandSample {
  landmarks: NormalizedLandmark[];
  /** Metric 3D coordinates (metres) centred on the hand — pose- and distance-invariant, unlike `landmarks`. */
  worldLandmarks: Landmark[];
}

function normalize(x: number, y: number, z: number): { x: number; y: number; z: number } {
  const len = Math.hypot(x, y, z) || 1e-6;
  return { x: x / len, y: y / len, z: z / len };
}

/**
 * Projection scale: how much of the frame a one-metre object spans at the
 * hand's current distance.
 *
 * This is the third attempt at a scale metric, and the first that actually
 * separates "hand moved away" from "hand rotated":
 *   1. A single wrist-to-middle-MCP distance — collapses when that one bone
 *      foreshortens.
 *   2. sqrt(palm polygon area) — no better, because a flat shape's projected
 *      *area* shrinks under any tilt too. Swapping one 2D measurement for
 *      another can't fix this: 2D projection alone genuinely cannot tell
 *      distance and rotation apart.
 *
 * The fix needs a rotation-invariant reference, which worldLandmarks
 * provides: metric 3D coordinates whose distances stay fixed no matter how
 * the hand turns. For each rigid palm segment, (2D projected length) /
 * (3D metric length) would be exactly the projection scale — if that
 * segment were broadside to the camera. A foreshortened segment yields a
 * smaller ratio instead. So take the *maximum* ratio across every palm
 * segment: rotating a hand foreshortens some of its segments, but the palm
 * spans enough directions that rarely all of them at once, and the
 * least-foreshortened one recovers the true scale.
 */
function projectionScale(landmarks: NormalizedLandmark[], worldLandmarks: Landmark[]): number {
  let maxRatio = 0;
  for (let i = 0; i < PALM_LANDMARKS.length; i++) {
    for (let j = i + 1; j < PALM_LANDMARKS.length; j++) {
      const a = PALM_LANDMARKS[i];
      const b = PALM_LANDMARKS[j];
      const projected = Math.hypot(landmarks[a].x - landmarks[b].x, landmarks[a].y - landmarks[b].y);
      const metric = Math.hypot(
        worldLandmarks[a].x - worldLandmarks[b].x,
        worldLandmarks[a].y - worldLandmarks[b].y,
        worldLandmarks[a].z - worldLandmarks[b].z,
      );
      if (metric > 1e-5) maxRatio = Math.max(maxRatio, projected / metric);
    }
  }
  return maxRatio;
}

/**
 * Derives the anchor point for a jutsu effect from one hand: palm centre
 * (position), a rotation-invariant scale (see projectionScale), and the
 * palm normal (which way the hand is facing — used to sit the effect on
 * top of the palm rather than buried inside it, and available for
 * orienting directional effects later).
 */
export function computeHandAnchor({ landmarks, worldLandmarks }: HandSample): HandAnchor {
  let x = 0, y = 0, z = 0;
  for (const i of PALM_LANDMARKS) {
    x += landmarks[i].x;
    y += landmarks[i].y;
    z += landmarks[i].z;
  }
  x /= PALM_LANDMARKS.length;
  y /= PALM_LANDMARKS.length;
  z /= PALM_LANDMARKS.length;

  // Palm normal via the cross product of two edges of the wrist/index-MCP/
  // pinky-MCP triangle — three points that stay roughly coplanar with the
  // palm regardless of finger pose.
  const wrist = landmarks[WRIST];
  const indexMcp = landmarks[INDEX_MCP];
  const pinkyMcp = landmarks[PINKY_MCP];
  const ax = indexMcp.x - wrist.x, ay = indexMcp.y - wrist.y, az = indexMcp.z - wrist.z;
  const bx = pinkyMcp.x - wrist.x, by = pinkyMcp.y - wrist.y, bz = pinkyMcp.z - wrist.z;
  let normal = normalize(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx);
  // The cross product's sign flips between left and right hands, which would
  // push the effect *into* the hand for one of them. MediaPipe's z grows
  // away from the camera, so forcing z <= 0 makes this consistently point
  // out of the palm toward the viewer for either hand.
  if (normal.z > 0) normal = { x: -normal.x, y: -normal.y, z: -normal.z };

  const scale = projectionScale(landmarks, worldLandmarks) * REFERENCE_PALM_METRES;

  return { x, y, z, scale, normal };
}

/** Picks which detected hand to anchor the effect to — the nearest one (largest projected palm), so a bystander further back can't steal the anchor. */
export function selectPrimaryHand(result: HandLandmarkerResult): HandSample | null {
  let best: HandSample | null = null;
  let bestSpan = -Infinity;
  for (let i = 0; i < result.landmarks.length; i++) {
    const landmarks = result.landmarks[i];
    const worldLandmarks = result.worldLandmarks[i];
    if (!landmarks || !worldLandmarks) continue;
    const span = Math.hypot(
      landmarks[INDEX_MCP].x - landmarks[PINKY_MCP].x,
      landmarks[INDEX_MCP].y - landmarks[PINKY_MCP].y,
    );
    if (span > bestSpan) {
      bestSpan = span;
      best = { landmarks, worldLandmarks };
    }
  }
  return best;
}
