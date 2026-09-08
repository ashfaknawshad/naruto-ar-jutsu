import type { HandLandmarkerResult, NormalizedLandmark } from "@mediapipe/tasks-vision";

const WRIST = 0;
const INDEX_MCP = 5;
const PINKY_MCP = 17;
// Palm centroid landmarks, per the design plan's anchoring approach.
const PALM_LANDMARKS = [0, 5, 9, 13, 17];

export interface HandAnchor {
  /** Palm centroid, in MediaPipe's normalized 0-1 image coordinates. */
  x: number;
  y: number;
  z: number;
  /** sqrt(palm polygon area), normalized — a proxy for hand size on screen, i.e. proximity to the camera. Bigger = closer. See computeHandAnchor's doc comment for why this isn't a single bone length. */
  scale: number;
  /** Unit palm-normal vector (image-space-relative, not camera-calibrated). */
  normal: { x: number; y: number; z: number };
}

function normalize(x: number, y: number, z: number): { x: number; y: number; z: number } {
  const len = Math.hypot(x, y, z) || 1e-6;
  return { x: x / len, y: y / len, z: z / len };
}

/** Shoelace formula. `points` must already trace the polygon's perimeter in order (not an arbitrary point set). */
function polygonArea2D(points: { x: number; y: number }[]): number {
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    const p1 = points[i];
    const p2 = points[(i + 1) % points.length];
    area += p1.x * p2.y - p2.x * p1.y;
  }
  return Math.abs(area) / 2;
}

/**
 * Derives the anchor point for a jutsu effect from one hand's landmarks:
 * palm centre (position), palm-polygon-area scale (a depth/scale proxy — no
 * depth camera needed, see the design plan §4.3), and the palm normal
 * (which way the hand is facing, for orienting directional effects later).
 *
 * Scale went through two versions before this one:
 *   1. A single wrist-to-middle-MCP distance — foreshortens badly under
 *      rotation, since one bone's 2D projection shrinks under a rotation
 *      that doesn't actually move the hand further away.
 *   2. sqrt(palm polygon area) over the wrist + 5 MCPs, matching how
 *      MediaPipe's own palm detector favours that rigid region over any
 *      single bone. This fixed in-plane foreshortening (e.g. cupping the
 *      palm so it still faces the camera) but not out-of-plane tilt —
 *      a flat shape's projected *area* shrinks under any tilt away from
 *      facing the camera, by basic projective geometry, regardless of
 *      whether you measure it as one edge or as area. Confirmed live: the
 *      effect still shrank hard when the palm turned to a more oblique
 *      angle, not just when cupped toward the camera.
 * This version corrects for that directly using the palm normal (computed
 * below) we already had sitting unused: dividing the raw projected scale
 * by how much the palm faces the camera (|normal.z|, clamped so a
 * near-edge-on hand — where this estimate is inherently unreliable anyway
 * — doesn't blow the scale up toward infinity) recovers an estimate of the
 * palm's true, untilted size.
 */
export function computeHandAnchor(landmarks: NormalizedLandmark[]): HandAnchor {
  let x = 0, y = 0, z = 0;
  for (const i of PALM_LANDMARKS) {
    x += landmarks[i].x;
    y += landmarks[i].y;
    z += landmarks[i].z;
  }
  x /= PALM_LANDMARKS.length;
  y /= PALM_LANDMARKS.length;
  z /= PALM_LANDMARKS.length;

  const wrist = landmarks[WRIST];
  const palmPoints = PALM_LANDMARKS.map((i) => landmarks[i]);
  const rawScale = Math.sqrt(polygonArea2D(palmPoints)) || 1e-6;

  // Palm normal via the cross product of two edges of the wrist/index-MCP/
  // pinky-MCP triangle — three points that stay roughly coplanar with the
  // palm regardless of finger pose.
  const indexMcp = landmarks[INDEX_MCP];
  const pinkyMcp = landmarks[PINKY_MCP];
  const ax = indexMcp.x - wrist.x, ay = indexMcp.y - wrist.y, az = indexMcp.z - wrist.z;
  const bx = pinkyMcp.x - wrist.x, by = pinkyMcp.y - wrist.y, bz = pinkyMcp.z - wrist.z;
  const normal = normalize(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx);

  const MIN_FACING = 0.35;
  const facing = Math.max(Math.abs(normal.z), MIN_FACING);
  const scale = rawScale / facing;

  return { x, y, z, scale, normal };
}

/** Picks which detected hand to anchor the effect to — the one with the largest palm area (closest to the camera), so a bystander in frame doesn't steal the anchor. */
export function selectPrimaryHand(result: HandLandmarkerResult): NormalizedLandmark[] | null {
  let best: NormalizedLandmark[] | null = null;
  let bestScale = -Infinity;
  for (const landmarks of result.landmarks) {
    const scale = polygonArea2D(PALM_LANDMARKS.map((i) => landmarks[i]));
    if (scale > bestScale) {
      bestScale = scale;
      best = landmarks;
    }
  }
  return best;
}
