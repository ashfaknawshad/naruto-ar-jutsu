import { WebGLRenderer, Scene, OrthographicCamera } from "three";
import { createRasengan } from "./rasengan";

export interface Stage {
  resize(width: number, height: number): void;
  /** Positions and scales the Rasengan from an anchor point (MediaPipe's normalized 0-1 image coords) and a size scale. z is accepted but unused until depth compositing (occlusion, Day 6+) needs it. */
  setAnchor(x: number, y: number, z: number, scale: number, visible: boolean): void;
  render(): void;
}

// Converts the wrist-to-MCP scale proxy (typically ~0.1-0.3 for a hand at
// conversational distance) into a world-space radius that roughly fills a
// palm. Eyeballed against the debug sphere in Day 4 testing.
const RADIUS_MULTIPLIER = 0.5;

/**
 * A minimal Three.js scene: an orthographic camera mapped 1:1 onto the
 * mirrored video/landmark canvases already in the DOM (see style.css's
 * scaleX(-1) trick — this canvas gets the same CSS transform, so raw
 * unmirrored MediaPipe coordinates just work here exactly like they do in
 * drawLandmarks.ts), holding the Rasengan effect.
 */
export function createStage(canvas: HTMLCanvasElement, width: number, height: number): Stage {
  const renderer = new WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new Scene();

  let aspect = width / height;
  // World y spans a fixed [-0.5, 0.5] ("fit height"); x scales by aspect —
  // keeps the effect circular instead of stretched on non-square frames.
  const camera = new OrthographicCamera(-aspect / 2, aspect / 2, 0.5, -0.5, 0.1, 10);
  camera.position.z = 5;

  const rasengan = createRasengan();
  rasengan.group.visible = false;
  scene.add(rasengan.group);

  const clockStart = performance.now();

  function resize(w: number, h: number) {
    renderer.setSize(w, h, false);
    aspect = w / h;
    camera.left = -aspect / 2;
    camera.right = aspect / 2;
    camera.updateProjectionMatrix();
  }
  resize(width, height);

  function setAnchor(x: number, y: number, _z: number, scale: number, visible: boolean) {
    rasengan.group.visible = visible;
    if (!visible) return;
    rasengan.group.position.set((x - 0.5) * aspect, -(y - 0.5), 0);
    rasengan.group.scale.setScalar(scale * RADIUS_MULTIPLIER);
  }

  function render() {
    if (rasengan.group.visible) rasengan.update((performance.now() - clockStart) / 1000);
    renderer.render(scene, camera);
  }

  return { resize, setAnchor, render };
}
