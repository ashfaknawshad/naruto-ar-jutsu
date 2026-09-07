import "./style.css";
import { initDevicePicker } from "./camera/devices";
import { initHandTracker } from "./perception/handTracker";
import { runLive } from "./modes/live";
import { runRecorder } from "./tools/recorder";

const video = document.querySelector<HTMLVideoElement>("#camera-feed")!;
const canvas = document.querySelector<HTMLCanvasElement>("#overlay")!;
const hud = document.querySelector<HTMLDivElement>("#hud")!;
const picker = document.querySelector<HTMLSelectElement>("#device-picker")!;
const ctx = canvas.getContext("2d")!;

// The display stays at full camera resolution, but MediaPipe only needs to
// see a small frame to find landmarks — inference cost scales with input
// pixel count, and 1280x720 into the model is most of our latency budget.
const DETECT_WIDTH = 480;
const detectCanvas = document.createElement("canvas");
const detectCtx = detectCanvas.getContext("2d", { willReadFrequently: true })!;

function resizeCanvasToVideo() {
  canvas.width = video.videoWidth || window.innerWidth;
  canvas.height = video.videoHeight || window.innerHeight;

  const aspect = canvas.height / canvas.width || 9 / 16;
  detectCanvas.width = DETECT_WIDTH;
  detectCanvas.height = Math.round(DETECT_WIDTH * aspect);
}

async function main() {
  hud.textContent = "requesting camera...";
  await initDevicePicker(picker, video, resizeCanvasToVideo);
  video.addEventListener("loadedmetadata", resizeCanvasToVideo);
  resizeCanvasToVideo();

  hud.textContent = "loading hand model...";
  await initHandTracker();

  const deps = { video, canvas, ctx, detectCanvas, detectCtx, hud };
  const mode = new URLSearchParams(location.search).get("mode");

  if (mode === "record") {
    runRecorder(deps);
  } else {
    runLive(deps);
  }
}

main().catch((err) => {
  hud.textContent = `error: ${err instanceof Error ? err.message : String(err)}`;
  console.error(err);
});
