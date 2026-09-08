import {
  WebGLRenderer,
  Scene,
  OrthographicCamera,
  AmbientLight,
  IcosahedronGeometry,
  MeshBasicMaterial,
  Mesh,
} from "three";

export interface Stage {
  resize(width: number, height: number): void;
  /** Positions and scales the debug sphere from an anchor point (MediaPipe's normalized 0-1 image coords) and a size scale. z is accepted but unused until depth compositing (occlusion, Day 6+) needs it. */
  setAnchor(x: number, y: number, z: number, scale: number, visible: boolean): void;
  render(): void;
}

// Tuning constant: converts the wrist-to-MCP scale proxy (typically
// ~0.1-0.3 for a hand at conversational distance) into a sphere radius
// that roughly covers a palm. Eyeballed, will get revisited once the real
// Rasengan shader replaces this debug placeholder in Day 5.
const RADIUS_MULTIPLIER = 1.4;

/**
 * A minimal Three.js scene: an orthographic camera mapped 1:1 onto the
 * mirrored video/landmark canvases already in the DOM (see style.css's
 * scaleX(-1) trick — this canvas gets the same CSS transform, so raw
 * unmirrored MediaPipe coordinates just work here exactly like they do in
 * drawLandmarks.ts), plus a wireframe sphere standing in for the eventual
 * VFX. Day 4's whole job is proving this sphere tracks a palm convincingly.
 */
export function createStage(canvas: HTMLCanvasElement, width: number, height: number): Stage {
  const renderer = new WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new Scene();

  let aspect = width / height;
  // World y spans a fixed [-0.5, 0.5] ("fit height"); x scales by aspect —
  // keeps the sphere circular instead of stretched on non-square frames.
  const camera = new OrthographicCamera(-aspect / 2, aspect / 2, 0.5, -0.5, 0.1, 10);
  camera.position.z = 5;

  scene.add(new AmbientLight(0xffffff, 1));

  const geometry = new IcosahedronGeometry(1, 1);
  const material = new MeshBasicMaterial({ color: 0x7dfcff, wireframe: true });
  const sphere = new Mesh(geometry, material);
  sphere.visible = false;
  scene.add(sphere);

  function resize(w: number, h: number) {
    renderer.setSize(w, h, false);
    aspect = w / h;
    camera.left = -aspect / 2;
    camera.right = aspect / 2;
    camera.updateProjectionMatrix();
  }
  resize(width, height);

  function setAnchor(x: number, y: number, _z: number, scale: number, visible: boolean) {
    sphere.visible = visible;
    if (!visible) return;
    sphere.position.set((x - 0.5) * aspect, -(y - 0.5), 0);
    sphere.scale.setScalar(scale * RADIUS_MULTIPLIER);
    sphere.rotation.y += 0.02; // idle spin — just a "this is live" cue for now
  }

  function render() {
    renderer.render(scene, camera);
  }

  return { resize, setAnchor, render };
}
