import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";

// Self-hosted so the booth (and this flaky library wifi) never depends on
// jsdelivr/googleapis being reachable at runtime.
const WASM_BASE = "/mediapipe/wasm";
const MODEL_URL = "/models/hand_landmarker.task";

let landmarker: HandLandmarker | null = null;

export async function initHandTracker(): Promise<HandLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  return landmarker;
}

/**
 * Detects hands in the given video frame. Never call this concurrently with
 * itself — the caller's render loop should skip a frame if the previous
 * detection hasn't returned yet rather than queueing calls.
 */
export function detectHands(video: HTMLVideoElement, timestampMs: number): HandLandmarkerResult {
  if (!landmarker) throw new Error("initHandTracker() must be awaited before detectHands()");
  return landmarker.detectForVideo(video, timestampMs);
}
