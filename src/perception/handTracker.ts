import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";

// Self-hosted so the booth (and this flaky library wifi) never depends on
// jsdelivr/googleapis being reachable at runtime. BASE_URL accounts for
// GitHub Pages serving the build from /naruto-ar-jutsu/ rather than root.
const WASM_BASE = `${import.meta.env.BASE_URL}mediapipe/wasm`;
const MODEL_URL = `${import.meta.env.BASE_URL}models/hand_landmarker.task`;

let landmarker: HandLandmarker | null = null;

/**
 * Wraps window.fetch for the duration of `fn` so downloads of the wasm
 * runtime + model (~19MB combined, on a fresh visit) report bytes received
 * as they stream in. Without this the loading screen sits on one static
 * message for however long the download takes, which reads as hung rather
 * than working — especially on mobile data.
 */
async function withFetchProgress<T>(onProgress: (loadedBytes: number) => void, fn: () => Promise<T>): Promise<T> {
  const originalFetch = window.fetch.bind(window);
  let loaded = 0;
  window.fetch = async (...args) => {
    const res = await originalFetch(...args);
    if (!res.body || !res.headers.get("content-length")) return res;
    const reader = res.body.getReader();
    const stream = new ReadableStream({
      async pull(controller) {
        const { done, value } = await reader.read();
        if (done) {
          controller.close();
          return;
        }
        loaded += value.byteLength;
        onProgress(loaded);
        controller.enqueue(value);
      },
    });
    return new Response(stream, { headers: res.headers, status: res.status, statusText: res.statusText });
  };
  try {
    return await fn();
  } finally {
    window.fetch = originalFetch;
  }
}

export async function initHandTracker(onProgress?: (loadedBytes: number) => void): Promise<HandLandmarker> {
  return withFetchProgress(onProgress ?? (() => {}), async () => {
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
  });
}

/**
 * Detects hands in the given frame (typically a downscaled canvas fed from
 * the camera, not the full-resolution video element — inference cost scales
 * with input pixels). Never call this concurrently with itself — the
 * caller's render loop should skip a frame if the previous detection hasn't
 * returned yet rather than queueing calls.
 */
export function detectHands(
  source: HTMLVideoElement | HTMLCanvasElement,
  timestampMs: number,
): HandLandmarkerResult {
  if (!landmarker) throw new Error("initHandTracker() must be awaited before detectHands()");
  return landmarker.detectForVideo(source, timestampMs);
}
