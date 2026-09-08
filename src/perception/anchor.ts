import type { HandLandmarkerResult, NormalizedLandmark } from "@mediapipe/tasks-vision";

const WRIST = 0;
const MIDDLE_MCP = 9;
const INDEX_MCP = 5;
const PINKY_MCP = 17;
// Palm centroid landmarks, per the design plan's anchoring approach.
const PALM_LANDMARKS = [0, 5, 9, 13, 17];

export interface HandAnchor {
  /** Palm centroid, in MediaPipe's normalized 0-1 image coordinates. */
  x: number;
  y: number;
  z: number;
  /** Wrist-to-middle-MCP span, normalized — a proxy for hand size on screen, i.e. proximity to the camera. Bigger = closer. */
  scale: number;
  /** Unit palm-normal vector (image-space-relative, not camera-calibrated). */
  normal: { x: number; y: number; z: number };
}

function normalize(x: number, y: number, z: number): { x: number; y: number; z: number } {
  const len = Math.hypot(x, y, z) || 1e-6;
  return { x: x / len, y: y / len, z: z / len };
}

/**
 * Derives the anchor point for a jutsu effect from one hand's landmarks:
 * palm centre (position), wrist-to-MCP span (a depth/scale proxy — no depth
 * camera needed, see the design plan §4.3), and the palm normal (which way
 * the hand is facing, for orienting directional effects later).
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
  const mcp = landmarks[MIDDLE_MCP];
  const scale = Math.hypot(mcp.x - wrist.x, mcp.y - wrist.y, mcp.z - wrist.z) || 1e-6;

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

/** Picks which detected hand to anchor the effect to — the one with the largest palm span (closest to the camera), so a bystander in frame doesn't steal the anchor. */
export function selectPrimaryHand(result: HandLandmarkerResult): NormalizedLandmark[] | null {
  let best: NormalizedLandmark[] | null = null;
  let bestScale = -Infinity;
  for (const landmarks of result.landmarks) {
    const wrist = landmarks[WRIST];
    const mcp = landmarks[MIDDLE_MCP];
    const scale = Math.hypot(mcp.x - wrist.x, mcp.y - wrist.y);
    if (scale > bestScale) {
      bestScale = scale;
      best = landmarks;
    }
  }
  return best;
}
