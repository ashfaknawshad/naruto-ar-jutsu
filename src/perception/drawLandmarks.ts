import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";

// 21-point MediaPipe hand connections.
const CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], // thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // index
  [5, 9], [9, 10], [10, 11], [11, 12], // middle
  [9, 13], [13, 14], [14, 15], [15, 16], // ring
  [13, 17], [17, 18], [18, 19], [19, 20], // pinky
  [0, 17], // palm base
];

export function drawHandLandmarks(
  ctx: CanvasRenderingContext2D,
  result: HandLandmarkerResult,
  width: number,
  height: number,
): void {
  ctx.clearRect(0, 0, width, height);

  for (let h = 0; h < result.landmarks.length; h++) {
    const landmarks = result.landmarks[h];
    const handedness = result.handedness[h]?.[0]?.categoryName ?? "?";
    const color = handedness === "Left" ? "#7dfcff" : "#ff9d7d";

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (const [a, b] of CONNECTIONS) {
      const pa = landmarks[a];
      const pb = landmarks[b];
      ctx.moveTo(pa.x * width, pa.y * height);
      ctx.lineTo(pb.x * width, pb.y * height);
    }
    ctx.stroke();

    ctx.fillStyle = color;
    for (const p of landmarks) {
      ctx.beginPath();
      ctx.arc(p.x * width, p.y * height, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
