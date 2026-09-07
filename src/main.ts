import "./style.css";
import { initDevicePicker } from "./camera/devices";
import { initHandTracker, detectHands } from "./perception/handTracker";
import { drawHandLandmarks } from "./perception/drawLandmarks";

const video = document.querySelector<HTMLVideoElement>("#camera-feed")!;
const canvas = document.querySelector<HTMLCanvasElement>("#overlay")!;
const hud = document.querySelector<HTMLDivElement>("#hud")!;
const picker = document.querySelector<HTMLSelectElement>("#device-picker")!;
const ctx = canvas.getContext("2d")!;

function resizeCanvasToVideo() {
  canvas.width = video.videoWidth || window.innerWidth;
  canvas.height = video.videoHeight || window.innerHeight;
}

async function main() {
  hud.textContent = "requesting camera...";
  await initDevicePicker(picker, video, resizeCanvasToVideo);
  video.addEventListener("loadedmetadata", resizeCanvasToVideo);
  resizeCanvasToVideo();

  hud.textContent = "loading hand model...";
  await initHandTracker();

  // Rolling FPS average and a busy-flag so a slow detection never queues —
  // we drop frames instead of falling behind, per the architecture plan.
  let detecting = false;
  let lastFrameTime = performance.now();
  const fpsWindow: number[] = [];
  let lastLatencyMs = 0;

  function loop() {
    const now = performance.now();
    const dt = now - lastFrameTime;
    lastFrameTime = now;
    fpsWindow.push(1000 / dt);
    if (fpsWindow.length > 30) fpsWindow.shift();
    const fps = fpsWindow.reduce((a, b) => a + b, 0) / fpsWindow.length;

    if (!detecting && video.readyState >= 2) {
      detecting = true;
      const t0 = performance.now();
      const result = detectHands(video, now);
      lastLatencyMs = performance.now() - t0;
      drawHandLandmarks(ctx, result, canvas.width, canvas.height);
      detecting = false;
    }

    hud.textContent = `fps: ${fps.toFixed(0)}\nhand detect: ${lastLatencyMs.toFixed(1)}ms\n${canvas.width}x${canvas.height}`;

    requestAnimationFrame(loop);
  }

  requestAnimationFrame(loop);
}

main().catch((err) => {
  hud.textContent = `error: ${err instanceof Error ? err.message : String(err)}`;
  console.error(err);
});
