import { detectHands } from "../perception/handTracker";
import { drawHandLandmarks } from "../perception/drawLandmarks";
import { buildFeatureVector, FEATURE_LENGTH } from "../perception/features";
import { ALL_LABELS, type Label } from "../data/seals";

const STORAGE_KEY = "kekkai-dataset-v1";
const MAX_FRAMES_PER_HOLD = 300; // ~10s safety cap in case a keyup is missed

interface Sample {
  label: Label;
  features: number[];
}

interface Dataset {
  version: 1;
  featureLength: number;
  samples: Sample[];
}

function loadDataset(): Dataset {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Dataset;
      if (parsed.featureLength === FEATURE_LENGTH) return parsed;
    }
  } catch {
    // Corrupt or missing — start fresh.
  }
  return { version: 1, featureLength: FEATURE_LENGTH, samples: [] };
}

function saveDataset(dataset: Dataset): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(dataset));
  } catch {
    // Quota exceeded — the Download button is the durable path; recording
    // can continue in memory even if the local backup stops updating.
  }
}

export interface RecorderDeps {
  video: HTMLVideoElement;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  detectCanvas: HTMLCanvasElement;
  detectCtx: CanvasRenderingContext2D;
  hud: HTMLDivElement;
}

export function runRecorder({ video, canvas, ctx, detectCanvas, detectCtx, hud }: RecorderDeps): void {
  const dataset = loadDataset();

  // --- UI ---------------------------------------------------------------
  const panel = document.createElement("div");
  panel.className = "recorder-panel";

  const labelSelect = document.createElement("select");
  labelSelect.className = "recorder-select";
  for (const label of ALL_LABELS) {
    const opt = document.createElement("option");
    opt.value = label;
    opt.textContent = label;
    labelSelect.appendChild(opt);
  }
  labelSelect.addEventListener("change", () => labelSelect.blur());

  const status = document.createElement("div");
  status.className = "recorder-status";

  const counts = document.createElement("div");
  counts.className = "recorder-counts";

  const buttons = document.createElement("div");
  buttons.className = "recorder-buttons";

  const downloadBtn = document.createElement("button");
  downloadBtn.textContent = "Download JSON";
  downloadBtn.addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(dataset)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `kekkai-dataset-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  const clearBtn = document.createElement("button");
  clearBtn.textContent = "Clear all";
  clearBtn.addEventListener("click", () => {
    if (!confirm(`Delete all ${dataset.samples.length} recorded samples?`)) return;
    dataset.samples.length = 0;
    saveDataset(dataset);
    renderCounts();
  });

  buttons.append(downloadBtn, clearBtn);
  panel.append(
    labelRow("Label:", labelSelect),
    status,
    counts,
    buttons,
  );
  document.querySelector("#app")!.appendChild(panel);

  function labelRow(text: string, control: HTMLElement): HTMLElement {
    const row = document.createElement("div");
    row.className = "recorder-row";
    const span = document.createElement("span");
    span.textContent = text;
    row.append(span, control);
    return row;
  }

  function renderCounts() {
    const tally = new Map<Label, number>();
    for (const s of dataset.samples) tally.set(s.label, (tally.get(s.label) ?? 0) + 1);
    counts.innerHTML = "";
    for (const label of ALL_LABELS) {
      const row = document.createElement("div");
      row.className = "recorder-count-row";
      row.textContent = `${label}: ${tally.get(label) ?? 0}`;
      counts.appendChild(row);
    }
    const total = document.createElement("div");
    total.className = "recorder-count-total";
    total.textContent = `total: ${dataset.samples.length}`;
    counts.appendChild(total);
  }
  renderCounts();

  // --- Recording ----------------------------------------------------------
  let recording = false;
  let framesThisHold = 0;

  window.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || e.repeat) return;
    if (document.activeElement === labelSelect) return;
    e.preventDefault();
    recording = true;
    framesThisHold = 0;
  });
  window.addEventListener("keyup", (e) => {
    if (e.code !== "Space") return;
    recording = false;
    saveDataset(dataset);
    renderCounts();
  });

  // --- Loop ---------------------------------------------------------------
  let detecting = false;
  let lastFrameTime = performance.now();
  const fpsWindow: number[] = [];

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
      const result = detectHands(detectCanvas, now);
      drawHandLandmarks(ctx, result, canvas.width, canvas.height);

      if (recording && framesThisHold < MAX_FRAMES_PER_HOLD) {
        const features = buildFeatureVector(result);
        dataset.samples.push({
          label: labelSelect.value as Label,
          features: Array.from(features),
        });
        framesThisHold++;
      } else if (recording) {
        recording = false; // hit the safety cap — force a fresh keypress
        saveDataset(dataset);
        renderCounts();
      }

      detecting = false;
    }

    status.textContent = recording
      ? `● recording "${labelSelect.value}" — ${framesThisHold} frames this hold`
      : "hold SPACE to record the selected label";
    status.classList.toggle("recording", recording);

    hud.textContent = `fps: ${fps.toFixed(0)}\nsamples: ${dataset.samples.length}`;

    requestAnimationFrame(loop);
  }

  requestAnimationFrame(loop);
}
