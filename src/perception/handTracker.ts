import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";

// TODO(day 10/11): download these into public/models/ and point at local
// paths so the booth doesn't depend on venue wifi.
const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

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
