import { detectHands } from "../perception/handTracker";
import { drawHandLandmarks } from "../perception/drawLandmarks";
import { buildFeatureVector, FEATURE_LENGTH } from "../perception/features";
import { ALL_LABELS, type Label } from "../data/seals";

const STORAGE_KEY = "kekkai-dataset-v1";
const RECORDER_NAME_KEY = "kekkai-recorder-name";
const COUNTDOWN_MS = 3000; // time to get both hands into position, hands-free
const CAPTURE_MS = 2000; // burst length once countdown hits zero
const REST_MS = 900; // pause between reps when auto-repeat is on

interface Sample {
  label: Label;
  features: number[];
  /** Who this rep was recorded from — needed to split train/test by person
   * rather than by frame (frame-level splits leak and inflate accuracy). */
  person: string;
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

type Phase = "idle" | "countdown" | "capturing" | "resting";

export function runRecorder({ video, canvas, ctx, detectCanvas, detectCtx, hud }: RecorderDeps): void {
  const dataset = loadDataset();

  // --- UI ---------------------------------------------------------------
  const panel = document.createElement("div");
  panel.className = "recorder-panel";

  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.placeholder = "your name";
  nameInput.className = "recorder-select";
  nameInput.value = localStorage.getItem(RECORDER_NAME_KEY) ?? "";
  nameInput.addEventListener("input", () => {
    localStorage.setItem(RECORDER_NAME_KEY, nameInput.value.trim());
  });

  const labelSelect = document.createElement("select");
  labelSelect.className = "recorder-select";
  for (const label of ALL_LABELS) {
    const opt = document.createElement("option");
    opt.value = label;
    opt.textContent = label;
    labelSelect.appendChild(opt);
  }
  labelSelect.addEventListener("change", () => labelSelect.blur());

  const loopToggle = document.createElement("input");
  loopToggle.type = "checkbox";
  loopToggle.id = "recorder-loop";
  const loopLabel = document.createElement("label");
  loopLabel.htmlFor = "recorder-loop";
  loopLabel.textContent = "auto-repeat";
  loopLabel.style.display = "flex";
  loopLabel.style.alignItems = "center";
  loopLabel.style.gap = "4px";
  loopLabel.prepend(loopToggle);

  const status = document.createElement("div");
  status.className = "recorder-status";

  const counts = document.createElement("div");
  counts.className = "recorder-counts";

  const buttons = document.createElement("div");
  buttons.className = "recorder-buttons";

  const startBtn = document.createElement("button");
  startBtn.textContent = "Start (Space)";
  startBtn.addEventListener("click", () => beginRep());

  const chartImg = document.createElement("img");
  chartImg.src = `${import.meta.env.BASE_URL}image.png`;
  chartImg.alt = "Reference chart of the 12 hand seals";
  chartImg.className = "reference-chart visible"; // shown by default — friends need this to know what to record

  const chartBtn = document.createElement("button");
  chartBtn.textContent = "Hide seal chart";
  chartBtn.addEventListener("click", () => {
    const visible = chartImg.classList.toggle("visible");
    chartBtn.textContent = visible ? "Hide seal chart" : "Show seal chart";
  });

  const undoBtn = document.createElement("button");
  undoBtn.textContent = "Undo last take (Backspace)";
  undoBtn.disabled = true;
  undoBtn.addEventListener("click", () => undoLastTake());

  const downloadBtn = document.createElement("button");
  downloadBtn.textContent = "Download JSON";
  downloadBtn.addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(dataset)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const who = nameInput.value.trim().replace(/[^a-z0-9]+/gi, "-").toLowerCase() || "unknown";
    a.download = `kekkai-dataset-${who}-${Date.now()}.json`;
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

  buttons.append(startBtn, chartBtn, undoBtn, downloadBtn, clearBtn);
  panel.append(
    labelRow("Name:", nameInput),
    labelRow("Label:", labelSelect),
    labelRow("", loopLabel),
    status,
    counts,
    buttons,
  );
  document.querySelector("#app")!.appendChild(panel);
  document.querySelector("#app")!.appendChild(chartImg);

  const countdownOverlay = document.createElement("div");
  countdownOverlay.className = "recorder-countdown";
  document.querySelector("#app")!.appendChild(countdownOverlay);

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
    const people = new Set(dataset.samples.map((s) => s.person));
    total.textContent = `total: ${dataset.samples.length} (${people.size} ${people.size === 1 ? "person" : "people"})`;
    counts.appendChild(total);
  }
  renderCounts();

  // --- Rep state machine ---------------------------------------------------
  // idle -> countdown (3s, hands-free) -> capturing (2s burst) -> resting
  // (0.9s) -> back to countdown if auto-repeat is on, else idle. Nothing
  // requires a key held during the actual sign, so two-handed seals work.
  let phase: Phase = "idle";
  let phaseEndsAt = 0;
  let framesThisRep = 0;
  let repStartIndex = 0;

  // The most recently completed take, so a bad rep (missed the pose, hand
  // out of frame, panned away) can be discarded without a full re-record.
  // Cleared once a new rep starts, so "undo" only ever means "the last one".
  let lastTake: { start: number; count: number; label: Label } | null = null;

  function setLastTake(take: typeof lastTake) {
    lastTake = take;
    undoBtn.disabled = !take;
  }

  let flashText = "";
  let flashUntil = 0;
  function flash(text: string, ms = 1500) {
    flashText = text;
    flashUntil = performance.now() + ms;
  }

  function undoLastTake() {
    if (!lastTake) return;
    dataset.samples.splice(lastTake.start, lastTake.count);
    saveDataset(dataset);
    renderCounts();
    flash(`discarded ${lastTake.count} frames of "${lastTake.label}"`);
    setLastTake(null);
  }

  function beginRep() {
    if (phase !== "idle" && phase !== "resting") return;
    if (!nameInput.value.trim()) {
      status.textContent = "enter your name first (top-left field) so samples can be split by person";
      nameInput.focus();
      return;
    }
    setLastTake(null);
    phase = "countdown";
    phaseEndsAt = performance.now() + COUNTDOWN_MS;
  }

  function cancelRep() {
    phase = "idle";
    countdownOverlay.textContent = "";
    countdownOverlay.classList.remove("visible");
  }

  window.addEventListener("keydown", (e) => {
    if (document.activeElement === labelSelect || document.activeElement === nameInput) return;
    if (e.code === "Space" && !e.repeat) {
      e.preventDefault();
      beginRep();
    } else if (e.code === "Escape") {
      cancelRep();
    } else if (e.code === "Backspace" && !e.repeat) {
      e.preventDefault();
      undoLastTake();
    }
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

    // Phase transitions are time-driven, independent of detection cadence.
    if (phase === "countdown" && now >= phaseEndsAt) {
      phase = "capturing";
      phaseEndsAt = now + CAPTURE_MS;
      framesThisRep = 0;
      repStartIndex = dataset.samples.length;
    } else if (phase === "capturing" && now >= phaseEndsAt) {
      saveDataset(dataset);
      renderCounts();
      setLastTake({ start: repStartIndex, count: framesThisRep, label: labelSelect.value as Label });
      if (loopToggle.checked) {
        phase = "resting";
        phaseEndsAt = now + REST_MS;
      } else {
        phase = "idle";
      }
    } else if (phase === "resting" && now >= phaseEndsAt) {
      phase = "countdown";
      phaseEndsAt = now + COUNTDOWN_MS;
    }

    if (!detecting && video.readyState >= 2) {
      detecting = true;
      detectCtx.drawImage(video, 0, 0, detectCanvas.width, detectCanvas.height);
      const result = detectHands(detectCanvas, now);
      drawHandLandmarks(ctx, result, canvas.width, canvas.height);

      if (phase === "capturing") {
        const features = buildFeatureVector(result);
        dataset.samples.push({
          label: labelSelect.value as Label,
          features: Array.from(features),
          person: nameInput.value.trim(),
        });
        framesThisRep++;
      }

      detecting = false;
    }

    updateOverlay(now);

    hud.textContent = `fps: ${fps.toFixed(0)}\nsamples: ${dataset.samples.length}`;

    requestAnimationFrame(loop);
  }

  function updateOverlay(now: number) {
    if (now < flashUntil) {
      status.textContent = flashText;
      status.classList.remove("recording");
      if (phase === "idle") {
        countdownOverlay.textContent = "";
        countdownOverlay.className = "recorder-countdown";
      }
      if (phase !== "countdown" && phase !== "capturing") return;
    }
    if (phase === "countdown") {
      const secsLeft = Math.ceil((phaseEndsAt - now) / 1000);
      countdownOverlay.textContent = String(Math.max(secsLeft, 1));
      countdownOverlay.className = "recorder-countdown visible";
      status.textContent = `get ready: "${labelSelect.value}"`;
      status.classList.remove("recording");
    } else if (phase === "capturing") {
      countdownOverlay.textContent = "●";
      countdownOverlay.className = "recorder-countdown visible recording";
      status.textContent = `● recording "${labelSelect.value}" — ${framesThisRep} frames`;
      status.classList.add("recording");
    } else if (phase === "resting") {
      countdownOverlay.textContent = "";
      countdownOverlay.className = "recorder-countdown";
      status.textContent = `captured ${framesThisRep} frames — next rep starting... (Backspace to undo)`;
      status.classList.remove("recording");
    } else {
      countdownOverlay.textContent = "";
      countdownOverlay.className = "recorder-countdown";
      status.textContent = lastTake
        ? `last take: ${lastTake.count} frames of "${lastTake.label}" — Backspace to undo, or Space for the next rep`
        : "press SPACE (or Start) — hands free until the countdown ends";
      status.classList.remove("recording");
    }
  }

  requestAnimationFrame(loop);
}
