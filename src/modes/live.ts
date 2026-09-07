import { detectHands } from "../perception/handTracker";
import { drawHandLandmarks } from "../perception/drawLandmarks";

export interface LiveDeps {
  video: HTMLVideoElement;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  detectCanvas: HTMLCanvasElement;
  detectCtx: CanvasRenderingContext2D;
  hud: HTMLDivElement;
}

export function runLive({ video, canvas, ctx, detectCanvas, detectCtx, hud }: LiveDeps): void {
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
      detectCtx.drawImage(video, 0, 0, detectCanvas.width, detectCanvas.height);
      const t0 = performance.now();
      const result = detectHands(detectCanvas, now);
      lastLatencyMs = performance.now() - t0;
      drawHandLandmarks(ctx, result, canvas.width, canvas.height);
      detecting = false;
    }

    hud.textContent = `fps: ${fps.toFixed(0)}\nhand detect: ${lastLatencyMs.toFixed(1)}ms\n${canvas.width}x${canvas.height}`;

    requestAnimationFrame(loop);
  }

  requestAnimationFrame(loop);
}
