import { WebGLRenderer, Scene, OrthographicCamera, Mesh, MeshBasicMaterial, ShapeGeometry, Shape, Vector2 } from "three";
import { createRasengan } from "./rasengan";
import { convexHull, type Point2D } from "../perception/convexHull";

export interface Stage {
  resize(width: number, height: number): void;
  /**
   * Positions and scales the Rasengan from an anchor point (MediaPipe's
   * normalized 0-1 image coords) and a size scale. z is accepted but
   * unused until real depth compositing needs it — occlusion (below) uses
   * a fixed fraction of the current radius instead, which is enough for a
   * flat 2D hand silhouette. `landmarks`, when given, rebuilds the
   * finger-occlusion mask from all 21 points (same 0-1 coordinate space);
   * omit it (or pass an empty array) to hide the mask.
   */
  setAnchor(x: number, y: number, z: number, scale: number, visible: boolean, landmarks?: Point2D[]): void;
  render(): void;
}

// Converts the wrist-to-MCP scale proxy (typically ~0.1-0.3 for a hand at
// conversational distance) into a world-space radius that roughly fills a
// palm. Eyeballed against the debug sphere in Day 4 testing.
const RADIUS_MULTIPLIER = 0.5;

// How far the hull is pushed outward from its own centroid — "slightly
// dilated" per the design plan, so the mask sits just past the fingers'
// actual edge rather than clipping into them.
const HULL_DILATION = 1.15;

/**
 * A minimal Three.js scene: an orthographic camera mapped 1:1 onto the
 * mirrored video/landmark canvases already in the DOM (see style.css's
 * scaleX(-1) trick — this canvas gets the same CSS transform, so raw
 * unmirrored MediaPipe coordinates just work here exactly like they do in
 * drawLandmarks.ts), holding the Rasengan effect and the hand-occlusion
 * mask that lets fingers appear in front of it.
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

  // Renders depth only (colorWrite: false) so it's itself invisible, but —
  // being an ordinary opaque material — Three.js draws it before the
  // Rasengan's transparent shells and writes it to the depth buffer, so
  // their fragments are correctly discarded wherever this mask is in
  // front. That's the whole occlusion trick: no per-pixel logic of our
  // own needed, just the renderer's default opaque-before-transparent
  // ordering doing its normal job.
  const occlusionMesh = new Mesh(new ShapeGeometry(new Shape()), new MeshBasicMaterial({ colorWrite: false }));
  occlusionMesh.visible = false;
  scene.add(occlusionMesh);

  const clockStart = performance.now();

  function resize(w: number, h: number) {
    renderer.setSize(w, h, false);
    aspect = w / h;
    camera.left = -aspect / 2;
    camera.right = aspect / 2;
    camera.updateProjectionMatrix();
  }
  resize(width, height);

  function toWorld(p: Point2D): Point2D {
    return { x: (p.x - 0.5) * aspect, y: -(p.y - 0.5) };
  }

  function updateOcclusionMask(landmarks: Point2D[], effectRadius: number) {
    const worldPoints = landmarks.map(toWorld);
    const hull = convexHull(worldPoints);
    if (hull.length < 3) {
      occlusionMesh.visible = false;
      return;
    }

    let cx = 0, cy = 0;
    for (const p of hull) {
      cx += p.x;
      cy += p.y;
    }
    cx /= hull.length;
    cy /= hull.length;
    const dilated = hull.map((p) => new Vector2(cx + (p.x - cx) * HULL_DILATION, cy + (p.y - cy) * HULL_DILATION));

    occlusionMesh.geometry.dispose();
    occlusionMesh.geometry = new ShapeGeometry(new Shape(dilated));
    // Cuts roughly the front ~40% (nearest-camera slice) of the sphere's
    // depth — occludes the near bulk fingers would realistically cover
    // without occluding the whole thing. Tuned by eye; may need
    // adjustment once seen live, same as the shader did.
    occlusionMesh.position.z = effectRadius * 0.4;
    occlusionMesh.visible = true;
  }

  function setAnchor(x: number, y: number, _z: number, scale: number, visible: boolean, landmarks?: Point2D[]) {
    rasengan.group.visible = visible;
    if (!visible) {
      occlusionMesh.visible = false;
      return;
    }
    rasengan.group.position.set((x - 0.5) * aspect, -(y - 0.5), 0);
    const effectRadius = scale * RADIUS_MULTIPLIER;
    rasengan.group.scale.setScalar(effectRadius);

    if (landmarks && landmarks.length >= 3) {
      updateOcclusionMask(landmarks, effectRadius);
    } else {
      occlusionMesh.visible = false;
    }
  }

  function render() {
    if (rasengan.group.visible) rasengan.update((performance.now() - clockStart) / 1000);
    renderer.render(scene, camera);
  }

  return { resize, setAnchor, render };
}
