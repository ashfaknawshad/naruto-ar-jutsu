import { detectHands } from "../perception/handTracker";
import { drawHandLandmarks } from "../perception/drawLandmarks";
import { buildFeatureVector } from "../perception/features";
import { classify } from "../perception/classifier";
import { SealSmoother } from "../perception/smoothing";
import { computeHandAnchor, selectPrimaryHand } from "../perception/anchor";
import { OneEuroVec3Filter, OneEuroFilter } from "../perception/oneEuro";
import { createStage } from "../vfx/stage";
import { createReferenceChart } from "../ui/referenceChart";

export interface LiveDeps {
  video: HTMLVideoElement;
  canvas: HTMLCanvasElement;
  stageCanvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  detectCanvas: HTMLCanvasElement;
  detectCtx: CanvasRenderingContext2D;
  hud: HTMLDivElement;
}

export function runLive({ video, canvas, stageCanvas, ctx, detectCanvas, detectCtx, hud }: LiveDeps): void {
  const smoother = new SealSmoother();
  const { toggleBtn } = createReferenceChart(false);
  toggleBtn.className = "reference-chart-toggle";
  document.querySelector("#app")!.appendChild(toggleBtn);

  // The stage's own resize() reads canvas.width/height (the 2D overlay,
  // which only main.ts ever sets) to decide its target size — never
  // stageCanvas's own buffer dimensions, which the renderer mutates itself
  // by applying devicePixelRatio. Reading that back would misread the
  // renderer's own write as "the video resized again" and trigger another
  // resize, compounding every frame into a runaway canvas size (see the
  // comment in main.ts's resizeCanvasToVideo for how this actually happened).
  const stage = createStage(stageCanvas, canvas.width, canvas.height);
  let stageWidth = canvas.width;
  let stageHeight = canvas.height;

  // Position gets heavier smoothing than scale — a jittery radius reads as
  // "breathing", which is far less distracting than a jittery position,
  // which reads as the effect not actually being attached to the hand.
  const positionFilter = new OneEuroVec3Filter(1.2, 0.6, 1.0);
  const scaleFilter = new OneEuroFilter(1.5, 0.3, 1.0);
  let framesSinceHandSeen = 0;
  const RESET_AFTER_MISSING_FRAMES = 15; // ~0.5s at 30fps — avoid a filter reset on every single dropped frame

  // Rolling FPS average and a busy-flag so a slow detection never queues —
  // we drop frames instead of falling behind, per the architecture plan.
  let detecting = false;
  let lastFrameTime = performance.now();
  const fpsWindow: number[] = [];
  let lastLatencyMs = 0;
  let heldText = "—";
  let anchorText = "no hand";

  function loop() {
    const now = performance.now();
    const dt = now - lastFrameTime;
    lastFrameTime = now;
    fpsWindow.push(1000 / dt);
    if (fpsWindow.length > 30) fpsWindow.shift();
    const fps = fpsWindow.reduce((a, b) => a + b, 0) / fpsWindow.length;

    if (canvas.width !== stageWidth || canvas.height !== stageHeight) {
      stageWidth = canvas.width;
      stageHeight = canvas.height;
      stage.resize(stageWidth, stageHeight);
    }

    if (!detecting && video.readyState >= 2) {
      detecting = true;
      detectCtx.drawImage(video, 0, 0, detectCanvas.width, detectCanvas.height);
      const t0 = performance.now();
      const result = detectHands(detectCanvas, now);
      lastLatencyMs = performance.now() - t0;
      drawHandLandmarks(ctx, result, canvas.width, canvas.height);

      const features = buildFeatureVector(result);
      const { label, confidence } = classify(features);
      const held = smoother.push(label, confidence);
      heldText = held ? `${held.label} (${(held.confidence * 100).toFixed(0)}%)` : "—";

      const primaryHand = selectPrimaryHand(result);
      if (primaryHand) {
        framesSinceHandSeen = 0;
        const anchor = computeHandAnchor(primaryHand);
        const [sx, sy, sz] = positionFilter.filter(anchor.x, anchor.y, anchor.z, now);
        const scale = scaleFilter.filter(anchor.scale, now);
        stage.setAnchor(sx, sy, sz, scale, true, primaryHand);
        anchorText = `raw(${anchor.x.toFixed(2)},${anchor.y.toFixed(2)}) filtered(${sx.toFixed(2)},${sy.toFixed(2)}) scale ${scale.toFixed(3)}`;
      } else {
        framesSinceHandSeen++;
        if (framesSinceHandSeen > RESET_AFTER_MISSING_FRAMES) {
          positionFilter.reset();
          scaleFilter.reset();
        }
        stage.setAnchor(0, 0, 0, 0, false);
        anchorText = "no hand";
      }

      detecting = false;
    }

    stage.render();

    hud.textContent = `fps: ${fps.toFixed(0)}\nhand detect: ${lastLatencyMs.toFixed(1)}ms\nseal: ${heldText}\nanchor: ${anchorText}`;

    requestAnimationFrame(loop);
  }

  requestAnimationFrame(loop);
}
