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
 * Scale used to be a single wrist-to-middle-MCP distance. That foreshortens
 * badly under rotation — cup your palm toward the camera (a real seal
 * pose) and that one bone's 2D projection shrinks even though the hand
 * hasn't moved further away, so the effect would shrink with it. MediaPipe
 * itself sidesteps exactly this by detecting palms via a bounding region
 * over the wrist + all 5 MCP knuckles rather than any single bone, because
 * that region is far more rigid under articulation. Following the same
 * idea: scale here is sqrt(palm polygon area) over those 5 points, not one
 * edge — a rotation that foreshortens one edge doesn't collapse the whole
 * polygon's area the same way, since the other edges span different
 * directions.
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
  const scale = Math.sqrt(polygonArea2D(palmPoints)) || 1e-6;

  // Palm normal via the cross product of two edges of the wrist/index-MCP/
  // pinky-MCP triangle — three points that stay roughly coplanar with the
  // palm regardless of finger pose.
  const indexMcp = landmarks[INDEX_MCP];
  const pinkyMcp = landmarks[PINKY_MCP];
  const ax = indexMcp.x - wrist.x, ay = indexMcp.y - wrist.y, az = indexMcp.z - wrist.z;
  const bx = pinkyMcp.x - wrist.x, by = pinkyMcp.y - wrist.y, bz = pinkyMcp.z - wrist.z;
  const normal = normalize(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx);

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
