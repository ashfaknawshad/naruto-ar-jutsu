# Kekkai — Solo Build Plan

Supersedes the scheduling/staffing sections of `naruto-jutsu-ar-project-plan.md`.
That document stays the design bible (visitor flow, jutsu catalogue, ethics, booth).
This one is how one person actually ships it.

---

## 1. Stack — frozen, no revisit

**Web. TypeScript + Vite + MediaPipe Tasks (JS) + Three.js. No backend.**

The original plan hedged between Python desktop (Option A) and web (Option B).
For a solo dev the hedge is the single biggest cost. Web wins outright here:

| Constraint | Why web wins |
|---|---|
| Must be quick | No packaging, no env hell, hot reload in ~80ms |
| Phone camera | Just open the URL on the phone — the phone *is* the camera |
| No compromise on VFX | WebGL2 shaders + additive blending are the same GPU the desktop stack would use |
| Lightweight | Ship procedural GLSL, not sprite sheets. Whole app < 1.5 MB + MediaPipe WASM |
| Solo testing | `git push` → GitHub Pages → test on any device in the room |

Dependencies, total:

```
three                      # renderer + shaders
@mediapipe/tasks-vision    # HandLandmarker, FaceLandmarker, ImageSegmenter
vite, typescript
@vitejs/plugin-basic-ssl   # HTTPS on LAN so the phone will grant camera access
```

That is it. **No TensorFlow.js, no postprocessing lib, no React.** The classifier is
~40 lines of matrix multiply. The bloom pass is ~60 lines. Both are lighter and
faster than the packages that would do them for you.

## 2. Camera setup for development

- **Daily dev:** laptop webcam. Fastest loop.
- **Phone testing:** `vite --host` with basic-ssl, open `https://<laptop-ip>:5173` on
  the phone, accept the cert warning once. Rear camera via
  `getUserMedia({ video: { facingMode: 'environment' } })`.
- **Phone as booth camera:** DroidCam / Camo exposes it as a virtual webcam; it then
  shows up in the in-app device picker like any other. Build the picker on day 1.

Lock exposure and white balance via `applyConstraints` where supported, fall back to
a manual brightness slider. Auto-exposure hunting is still the #1 detection killer.

---

## 3. Architecture

```
src/
  main.ts                 bootstrap + single rAF loop
  camera/devices.ts       enumerateDevices, picker, constraint locking
  perception/
    handTracker.ts        HandLandmarker wrapper, GPU delegate, VIDEO mode
    features.ts           normalisation to fixed-length vector
    classifier.ts         MLP forward pass, weights loaded from JSON
    smoothing.ts          sliding-window vote + hysteresis (0.85 in / 0.60 hold)
    oneEuro.ts            One Euro filter (scalar + vec3)
    anchor.ts             palm centre, depth from MCP span, palm normal, thrust velocity
  state/
    jutsuMachine.ts       idle -> charging -> active -> releasing
    sequences.ts          multi-seal chains with per-step timeout + transition class
  vfx/
    stage.ts              Three scene, video plate as background quad
    handMask.ts           convex hull to depth-only mask (occlusion)
    rasengan.ts           layered fbm sphere shader
    particles.ts          one THREE.Points, curl-noise motion in the vertex shader
    post.ts               bloom, chromatic aberration, radial blur, shake, flash
  ui/
    hud.ts tutorial.ts hints.ts capture.ts attract.ts
  tools/recorder.ts       data capture mode, reached via ?mode=record
train/train_mlp.py        numpy/sklearn to weights.json
public/models/            .task files + weights.json
```

**Threading.** Browsers give you this nearly free: MediaPipe runs on the GPU delegate,
`detectForVideo` is called once per rAF with the frame timestamp, and if it returns
late you render the last known landmarks through the One Euro filter rather than
stalling. Never await perception before drawing the video plate.

---

## 4. VFX — quality without weight

The original plan led with pre-rendered sprite sequences. **Drop that.** It assumed a
dedicated artist. Solo, it means megabytes of PNGs you have to author in Blender.
Procedural shaders are both prettier and smaller here.

**Rasengan** — three nested icosphere shells, additive:

1. Core: bright near-white, fresnel-boosted, slow rotation.
2. Mid: animated 3-octave fbm on a shared 256x256 noise texture, domain-warped by
   time, blue-cyan ramp. This is the swirl.
3. Outer: high-frequency noise, low alpha, counter-rotating — the volatile edge.

Plus a fresnel rim on the outermost shell so it reads spherical instead of flat.
Total cost ~0.4ms at 512px. This is the "no compromise" bit — spend your shader
budget here and nowhere else.

**Particles** — a single `THREE.Points` with 2000 vertices. Motion is entirely in the
vertex shader: curl noise around a `uPalm` uniform, per-particle phase from an
attribute. The CPU writes one vec3 per frame. Additive, no depth write.

**Occlusion** — build the convex hull of the 21 landmarks, dilate ~8px, render it as a
depth-only invisible mesh *before* the sphere. Fingers then cut the sphere body.
Then draw the glow layer with `depthTest: false` on top of everything — real light
bleeds over occluders, and that one line is what sells the whole effect.

**Bloom** — downsample to quarter res, two-pass separable blur, additive composite.
~1ms. Do not install a post-processing library for this.

**Release** — chromatic aberration + radial blur + screen shake + white flash, all
ramped over ~250ms then off. Transient cost, permanent impression.

**Rule:** additive blending for every energy effect, always. Dark backdrop at the booth.

---

## 5. Data — the solo shortcut

You cannot recruit 15 volunteers this month. The original coverage targets assumed a
team running three collection sessions. Replace with:

1. **Record yourself thoroughly.** Recorder tool, 13 classes (12 seals + `none`),
   ~400 frames each, across 4 distances x 3 lighting conditions x sleeves/bare/watch.
   Roughly 90 focused minutes.
2. **Augment x8 on the landmark vectors** — mirror L/R, rotate ±15°, scale 0.8–1.2,
   Gaussian jitter σ=0.01. Cheap, and it does most of the work a bigger cohort would.
3. **Hard negatives are still non-negotiable.** Record `none` while waving, clapping,
   pointing, holding a phone, adjusting glasses, arms crossed, and near-miss seals.
   Make this the largest class. Nearly all booth false-triggers come from ordinary
   movement, not from wrong seals.
4. **Borrow 3–4 people for 20 minutes each, test set only.** Never train on them.
   This preserves the held-out-person metric the report needs, at 5% of the cost.

Store landmark vectors only, never images. Dataset stays a few MB, retrains in
seconds, and the privacy section of the main plan holds unchanged.

**Generous thresholds plus operator override cover the accuracy gap.** A model at 88%
with a good override feels better at a booth than 94% with none.

---

## 6. Schedule — ~12 focused days

Each day is one sitting, and each ends with something visible on screen. Do not start
the next until the current one is demoable.

| Day | Deliverable | Done when |
|---|---|---|
| 1 | Vite skeleton, camera + device picker, HandLandmarker, landmarks drawn on video, FPS/latency HUD | 60fps with skeleton on your hands, on laptop and phone |
| 2 | `features.ts` normalisation + recorder mode (`?mode=record`) | A keypress records N labelled frames to a downloadable JSON |
| 3 | Record the dataset, `train_mlp.py`, `classifier.ts`, smoothing + hysteresis | Live seal name on screen, stable, no flicker at boundaries |
| 4 | Three stage, video plate, anchor solver (palm centre / depth / normal), One Euro | A debug wireframe sphere sits in your palm and tracks convincingly |
| 5 | **Rasengan shader** — the three shells, noise, fresnel | It looks genuinely good standing still. Take the time here. |
| 6 | Occlusion mask, particles, charge ring meter, thrust detection + launch | Full charge → hold → throw loop works |
| 7 | Post FX, audio (hum, crackle, peak), photo capture | **Vertical slice gate.** If this is not pretty, cut everything below. |
| 8 | Sharingan (FaceLandmarker) + Chakra Aura (ImageSegmenter) | Two one-gesture effects kids cannot fail at |
| 9 | Chidori + sequence FSM + tutorial panel with live hand thumbnail + hints | 3-seal chain with "2/3 — next: Monkey" on screen |
| 10 | Attract mode, operator hotkeys, person lock-on, perf pass | 30 min unattended without degrading |
| 11 | Deploy to GitHub Pages, test on 3 different phones, fix what breaks | Public URL works from a QR code |
| 12 | Buffer | — |

Stretch, only after day 12: Shadow Clone (segmentation + smoke), Fireball (6-seal),
leaderboard, two-player duel.

**Cut list, in order, if you fall behind:** Fireball → Shadow Clone → Chidori →
sequence FSM. Never cut: Rasengan polish, Sharingan, operator override, attract mode.

---

## 7. What carries over unchanged

Sections 2 (visitor experience), 3 (catalogue + accessibility rule), 4.2–4.3
(recognition and anchoring — the technical core is correct as written), 7 (booth),
8 (reliability), 11 (educational layer), 13 (ethics), 15 (definition of done).

## 8. What changes for solo

- Section 4.4: web stack only; procedural shaders replace pre-rendered sprites.
- Section 5: reduced cohort plus heavier augmentation; borrowed test set.
- Section 9: 12 weeks becomes 12 focused days.
- Section 10 (roles): all of them are you. Order of attention: tracking feel > VFX
  quality > model accuracy. The first two are what a visitor actually perceives.
- Watchdog (section 8) becomes a page-visibility plus error-boundary auto-reload,
  ~15 lines in a browser instead of an external supervisor process.
