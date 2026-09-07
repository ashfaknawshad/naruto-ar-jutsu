# Project "Kekkai" — Naruto Hand Sign Recognition with Real-Time AR Jutsu Effects

**Team:** University of Moratuwa
**Audience:** School students and general public at a project showcase
**Nature:** Live interactive computer vision installation

---

## 1. Assumptions

These are the assumptions this plan is built on. Change them and some estimates shift.

- Team of 4–6 people, roughly 10–12 weeks of part-time work.
- One booth, one camera, one large display, an operator present at all times.
- Visitors range from 8-year-olds to adults. Many will have never seen a CV demo.
- Each visitor gets 60–90 seconds. Queue will form.
- Budget is modest. Existing laptops, one good webcam, borrowed lighting.

---

## 2. What the visitor actually experiences

This is the product. Everything technical serves this.

**Idle (attract mode).** Nobody in front of the camera. The screen loops a short highlight reel of the effects plus the words "Step on the mark and form the seal." Audio ambience plays quietly. This pulls people in from across the hall.

**Detection.** A person steps onto a taped floor marker. The system detects a body, greets them, and switches to live mirrored video. A hand skeleton overlay appears on their hands so they immediately understand the machine is watching their fingers.

**Guided tutorial (10 seconds).** A panel on the right shows the seal to make, as a large illustrated silhouette plus a live thumbnail of their own hands. This side-by-side is critical. People fix their own hand position when they can see the target and themselves at once.

**Charge.** They form the seal. A ring meter fills around their hands over 2–3 seconds. A rising audio hum plays. Particles start drifting inward toward the palm. If the seal breaks, the meter drains and a hint appears ("Keep your palms together").

**Manifestation.** The jutsu appears, anchored to the hand. A Rasengan sphere sits in the palm, spinning, casting light on the surroundings. As the hand moves, the sphere moves with it, and it scales as the hand comes closer to or further from the camera. Fingers occlude the front edge of the sphere so it reads as held, not pasted on.

**Release.** A forward thrust of the palm launches it. The sphere shrinks and fades into the distance, the screen shakes, a flash fires, and the audio peaks.

**Capture and exit.** A photo is auto-taken at the peak frame. A QR code appears for a few seconds so they can download it. The operator waves the next person forward.

**Progression.** Repeat visitors get a harder mode: multi-seal sequences and a timed leaderboard.

---

## 3. Jutsu catalogue

Design the content first, then build only what the content needs.

| Jutsu | Trigger | Effect | Difficulty to build |
|---|---|---|---|
| **Rasengan** | Hold open palm / cupped palm for 3s | Spinning blue sphere anchored to palm, follows hand, launches on thrust | Medium — the flagship |
| **Chidori** | Ox → Hare → Monkey sequence | Lightning arcs from hand, screen flicker, crackle audio, arcs jump to nearby edges | Medium |
| **Fireball (Katon)** | Snake → Ram → Monkey → Boar → Horse → Tiger, then hand to mouth | Fireball erupts toward camera, warm light wash | Hard (6-seal sequence) |
| **Shadow Clone** | Cross seal (two fingers crossed) held 2s | Person is segmented and duplicated 2–4 times across the frame with a puff of smoke | Medium, very high wow factor |
| **Sharingan** | Face detected + Tiger seal, or a simple button for young kids | Animated Sharingan iris overlaid on the eyes via face mesh | Easy, huge crowd appeal |
| **Chakra aura** | Ram seal held | Body-outline glow using segmentation, ground dust particles | Easy, good filler |
| **Susanoo** (stretch) | Long combo | Large skeletal figure aligned to body pose behind the person | Hard, stretch goal only |

**Design rule:** ship Rasengan, Sharingan and Chakra Aura first as a complete vertical slice. Everything else is additive. A booth with three polished effects beats a booth with seven broken ones.

**Accessibility rule:** small children cannot form two-handed seals reliably. Sharingan and Chakra Aura must be triggerable with one simple gesture so nobody walks away having failed.

---

## 4. Technical architecture

### 4.1 Pipeline overview

```
Camera (1080p, 60fps)
   ↓
Capture thread → ring buffer (drop-oldest)
   ↓
Perception thread
   ├── Hand landmarks (21 pts × 2 hands, 3D)
   ├── Pose landmarks (for body anchoring, person selection)
   ├── Face mesh (Sharingan only, on demand)
   └── Person segmentation mask (clones, aura, occlusion)
   ↓
Feature normalisation → Seal classifier (13 classes)
   ↓
Temporal smoothing + State machine (idle / charging / active / releasing)
   ↓
Anchor solver (position, scale, orientation, depth) + One Euro filter
   ↓
Render thread — video plate + VFX layers + UI + audio
   ↓
Display (mirrored) + photo capture
```

Threads are decoupled with bounded queues. If perception falls behind, frames are dropped rather than queued, so the video never lags behind the person's movement. This matters more than raw accuracy for perceived quality.

### 4.2 Hand sign recognition

**Do not train a CNN on raw images.** It needs far more data, is brittle to lighting and backgrounds, and is slower. Use landmarks.

**Step 1 — Landmark extraction.** MediaPipe Hands (or MediaPipe Tasks HandLandmarker) gives 21 3D landmarks per hand at high frame rate on CPU.

**Step 2 — Normalisation.** This is where accuracy is won or lost. For each frame:
- Translate so the wrist is at origin.
- Scale by the wrist-to-middle-finger-MCP distance, so the classifier is distance-invariant.
- Rotate to a canonical orientation so a tilted Tiger seal is still a Tiger seal.
- Concatenate both hands into a fixed-length vector. Add inter-hand features: distance between wrists, relative rotation, fingertip-to-fingertip distances across hands. Seals are defined by how the hands relate to each other, so these cross-hand features carry most of the signal.

**Step 3 — Classifier.** A small MLP (2 hidden layers, 128 and 64 units) or a Random Forest. Both train in seconds, run in under a millisecond, and are easy to retrain when you collect more data at 11pm the night before the showcase. Output: softmax over 12 seals + `none` + `transition`.

**Step 4 — Temporal smoothing.** Never act on a single frame. Maintain a sliding window of the last 10 frames. A seal is "held" only when it holds the majority with confidence above a threshold. Use hysteresis: entering a state needs confidence 0.85, staying in it only needs 0.60. Without hysteresis, the effect flickers at the boundary and looks broken.

**Step 5 — Sequence matching.** For multi-seal jutsu, a simple finite state machine. Track the seal sequence with a per-step timeout of 3 seconds. Show progress on screen ("2 / 6 — next: Monkey"). Allow a `transition` class between seals so hand movement between poses does not reset the chain.

### 4.3 Anchoring the effect to the hand

This is the part that makes it feel like an object rather than a sticker.

**Position.** Anchor to the palm centre, computed as the centroid of landmarks 0, 5, 9, 13, 17. For two-handed jutsu, use the midpoint between both palms.

**Scale and depth.** Estimate depth from apparent hand size. Measure the wrist-to-MCP distance in pixels and compare it to a calibrated reference. Closer hand means bigger pixel span means bigger sphere. This is a good approximation and needs no depth camera.

**Orientation.** Derive a palm normal vector from three non-collinear palm landmarks. Use it to tilt the effect so lightning arcs and directional effects point out of the hand correctly.

**Smoothing.** Use a **One Euro filter**, not a plain moving average. It gives low jitter when the hand is still and low lag when the hand moves fast. A moving average gives you one or the other, never both. This single choice is the difference between an effect that feels attached and one that feels like it is dragging behind.

**Occlusion.** Build a hand mask from the convex hull of the landmarks, slightly dilated. Render the sphere body behind that mask so fingers appear in front of it, then composite an additive glow layer on top of everything. The glow-on-top step is what sells it, because real light bleeds over the objects in front of it.

**Release detection.** Watch the derivative of the depth estimate and the palm normal. A rapid increase in apparent hand size plus the palm normal turning toward the camera equals a thrust. Threshold on velocity, not position.

### 4.4 Rendering and VFX

Two viable stacks. Pick one in week 2 and commit.

**Option A — Python desktop (recommended for the booth).**
OpenCV for capture and compositing, MediaPipe for perception, and ModernGL or Pygame for GPU-accelerated particles and shaders. Full control, no browser sandbox, easy to use a dedicated GPU.

**Option B — Web (recommended as a takeaway).**
MediaPipe Tasks for JS plus Three.js. Runs in any browser with zero install. Put a QR code on the booth wall so visitors can try a lightweight version on their own phones after they leave. This dramatically extends the reach of the project beyond the queue.

Ideal outcome: Option A for the main booth, a cut-down Option B for phones.

**VFX approach.** Combine three layers:
1. **Pre-rendered sprite sequences** authored in Blender or After Effects, played back as PNG sequences with alpha. Best quality per unit of effort.
2. **Procedural particles** for the drifting chakra motes, so they respond live to hand movement.
3. **Screen-space post effects** on release: bloom, chromatic aberration, radial blur, screen shake, colour grade shift.

**Use additive blending for all energy effects.** Energy does not occlude what is behind it, it adds light to it. Alpha blending makes a Rasengan look like a flat blue circle. Additive makes it glow.

**Dark backdrop.** A matte black or deep navy cloth behind the visitor makes bloom and additive effects read far better, and improves segmentation quality for the clone effect. This is the cheapest visual upgrade available to you.

---

## 5. Dataset plan

Most student CV projects fail here, not in the modelling.

**What to store.** Store the extracted 42-point landmark vectors, not the images. The dataset becomes a few megabytes, retraining takes seconds, and you sidestep the privacy problem of holding footage of volunteers entirely.

**Build a recording tool in week 3.** A small utility where you press a key, hold a seal, and it records N frames labelled with that class. Without this tool, data collection is unbearable and the team will cut corners.

**Coverage targets.** Aim for 300–500 samples per class per condition axis:

- **People:** at least 12–15 different individuals. Hand size varies enormously between an adult and a 9-year-old.
- **Lighting:** bright, dim, warm indoor, side-lit, backlit.
- **Distance:** 1.0m, 1.5m, 2.0m, 2.5m.
- **Sleeves:** bare arms, long sleeves, wristwatch, bracelets.
- **Skin tone:** deliberately recruit a diverse set of volunteers. MediaPipe is generally robust here but verify it rather than assume.
- **Backgrounds:** cluttered and plain.

**Hard negatives matter as much as positives.** Record a large `none` class containing: waving, clapping, pointing, holding a phone, adjusting glasses, arms crossed, hands in pockets, and near-miss seals that are almost but not quite correct. Most false triggers at a live booth come from ordinary hand movement, not from wrong seals.

**Split by person, not by frame.** If frames from the same person appear in both train and test, your reported accuracy will be inflated by 15–20 percentage points and will collapse at the booth. Hold out 3 people entirely as the test set.

**Augmentation.** On the landmark vectors: small rotations, mirroring left/right, Gaussian jitter on coordinates, random scaling. Cheap and effective.

---

## 6. Evaluation plan

For the academic report and for knowing when you are actually done.

**Model metrics**
- Per-class precision, recall, F1. Publish the confusion matrix. Ram and Snake are visually similar and will be the hard pair.
- Held-out-person accuracy. Target above 92%.
- False activation rate: run the system against 30 minutes of idle footage of people milling around and count spurious triggers. Target under 1 per 5 minutes.

**System metrics**
- End-to-end latency from photon to pixel. Target under 100ms. Above 150ms the effect visibly lags the hand.
- Sustained frame rate. Target 30fps minimum, 60fps ideal.
- Uptime across a 4-hour session with no crash or restart.

**Human metrics (the ones that predict showcase success)**
- Time-to-first-success for a first-time user with no verbal help. Target under 20 seconds.
- Completion rate for under-12s versus adults. If the gap is large, the tutorial UI needs work, not the model.
- Unprompted repeat attempts. People trying again without being asked is the strongest signal you have something good.

---

## 7. Booth and physical setup

| Item | Notes |
|---|---|
| Laptop | Dedicated GPU strongly preferred. Test the CPU-only fallback anyway. |
| Second laptop | Hot spare, fully configured and tested. Non-negotiable. |
| Webcam | 1080p60 with manual exposure lock. Auto-exposure hunting under changing hall lighting will wreck detection. |
| Tripod | Camera at roughly chest height for an adult, so kids are still fully framed. |
| Display | 43-inch TV minimum, or a projector. Bigger is better for drawing a crowd. |
| Lighting | Two soft key lights from the front. Kill any window or spotlight behind the visitor. |
| Backdrop | Dark matte cloth, 2m wide. |
| Audio | Powered speakers. Test volume against hall noise before the day. |
| Floor marker | Bright tape "X" at the correct distance. Solves half your framing problems for free. |
| Wall poster | Large printed chart of all 12 seals so people in the queue learn while they wait. This turns dead queue time into engagement and shortens each turn. |
| Cables | Gaffer tape all runs. Power strip, extension, spare USB cable. |

**Camera settings:** lock exposure, lock white balance, disable autofocus. Set these once at the venue during setup, not in the lab.

---

## 8. Reliability and failure handling

Assume something will break in front of an audience.

- **Operator override.** Hidden keyboard shortcuts to trigger any effect manually. If the model misfires for a nervous child, the operator fires it anyway and the child leaves happy. Ship this.
- **Confidence hints instead of silence.** If confidence sits between 0.4 and 0.8, show a targeted hint rather than nothing: "Move closer", "Bring your hands together", "Your left hand is out of frame". Silence reads as a broken machine.
- **Watchdog.** External process that detects a hang or crash and relaunches within 5 seconds into attract mode.
- **Camera loss fallback.** If the camera drops out, immediately switch to the pre-recorded showreel with a small "reconnecting" indicator, rather than showing an error.
- **Person selection.** With a crowd in frame, select the single person with the largest bounding box nearest the floor marker and lock onto them until they leave. Otherwise the system will flip between visitors and never charge.
- **Thermal.** Four hours of continuous GPU load will throttle a laptop. Test a full-length run beforehand and add a cooling pad if needed.

---

## 9. Timeline (12 weeks)

**Weeks 1–2 — Foundation**
Stack decision (A or B). MediaPipe pipeline running end to end at target frame rate. Draw the raw landmarks on the video and confirm performance on the actual booth laptop. Decide the jutsu catalogue and freeze it.

**Weeks 3–4 — Data**
Build the recording tool. Run three collection sessions with volunteers. Reach coverage targets including hard negatives. Begin VFX asset production in parallel — the artist should not be blocked waiting for the model.

**Week 5 — Model**
Feature normalisation, train and evaluate the classifier, tune the temporal smoothing and hysteresis. Produce the confusion matrix. Fix the worst-confused pair with targeted extra data.

**Weeks 6–7 — Vertical slice**
One complete jutsu: Rasengan, from seal through charge, anchoring, One Euro smoothing, occlusion, release, audio, and photo capture. **This is the critical milestone.** If Rasengan is not fully working and pretty by end of week 7, cut scope immediately.

**Weeks 8–9 — Content and polish**
Remaining jutsu. Segmentation for clones and aura. Sharingan face overlay. Full UI: tutorial panel, charge ring, hints, attract mode. Sound design pass.

**Week 10 — Integration and optimisation**
Thread the pipeline properly. Profile and hit the latency and frame rate targets. Watchdog, operator overrides, all fallbacks.

**Week 11 — User testing**
Test with actual school students who have never seen it. Watch silently and note every point of confusion. Do not explain. Whatever they struggle with is a UI defect. Iterate.

**Week 12 — Booth build and dry run**
Full four-hour rehearsal with the real hardware in a similar space. Print materials. Pack a kit list. Leave three days of pure buffer.

---

## 10. Roles

- **ML lead** — dataset design, classifier, evaluation, the report's quantitative sections.
- **CV / tracking engineer** — MediaPipe integration, normalisation, anchoring, filtering, occlusion.
- **VFX / graphics** — sprite production, particle systems, shaders, compositing, colour.
- **UX / frontend** — tutorial panel, hints, attract mode, photo and QR flow, audio.
- **Integration / ops** — threading, performance, watchdog, hardware, booth setup, dry runs.
- **Content / outreach** — seal poster, explainer boards, social clips, operator script.

Roles overlap in a small team. What matters is that someone owns each one by name.

---

## 11. The educational layer

The showcase is for school students, so the project should teach, not just impress. Budget a small second screen or a set of printed boards covering:

- **How the machine sees hands.** A live view showing only the 21-point skeleton on black. Genuinely mesmerising, and it makes the abstraction concrete.
- **Why landmarks and not pixels.** A one-panel explanation of the normalisation idea: the computer does not see fingers, it sees numbers describing where they are relative to each other.
- **What training data is.** Show the confusion matrix. Explain that Ram and Snake get confused because they look similar, and that fixing it meant showing the computer more examples. This is the single most useful idea a school student can take home about ML.
- **Where else this is used.** Sign language translation, surgical interfaces, accessibility tools, automotive gesture control. Connect the anime to real careers.

Have the operator ask each visitor one question: "How do you think it knew?" It converts a queue into a conversation.

---

## 12. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Hall lighting ruins detection | Severe | Own lighting, exposure lock, collect data under bad lighting deliberately, on-site calibration step |
| Kids cannot form seals | High | Simplified one-hand triggers, generous thresholds in "easy mode", operator override |
| Latency makes effects feel detached | High | One Euro filter, frame dropping, thread decoupling, profile from week 6 not week 11 |
| Scope creep across 7 jutsu | High | Vertical slice gate at week 7, freeze catalogue at week 2, ranked cut list agreed in advance |
| Model overfits to team members' hands | High | Split by person, recruit 12+ external volunteers, test on held-out people only |
| Crash during showcase | Medium | Watchdog, hot spare laptop, showreel fallback |
| Queue moves too slowly | Medium | 90-second hard cap, seal poster for queue learning, operator manages flow |
| Background crowd confuses tracking | Medium | Person lock-on, backdrop, floor marker, narrow camera FOV |

---

## 13. Ethics and privacy

Worth doing properly, and worth putting on a sign because it demonstrates maturity to the judges.

- No video or images are written to disk by default. Only landmark coordinates are used, and only in memory.
- The training dataset contains numeric landmarks only, no faces or images, collected from informed volunteers.
- Photo capture is opt-in, shown on screen before it happens, deleted after the QR download window expires.
- For visitors under 18, take photos only with an accompanying adult's agreement.
- Display a clear sign at the booth stating what the camera does and does not record.

---

## 14. Stretch goals

Only after the core is solid and rehearsed.

- **Two-player duel.** Two people, split frame, competing jutsu, collision between the effects.
- **Speed leaderboard.** Fastest correct six-seal Fireball sequence, displayed on a side screen.
- **Phone web version.** QR code takeaway using the browser stack.
- **Susanoo.** Full-body skeletal avatar aligned to pose landmarks.
- **Chakra nature quiz.** A short quiz assigns each visitor an element, and the booth then themes their effect accordingly. Cheap to build, strongly increases the sense of personalisation.

---

## 15. Definition of done

The project is ready when all of these are true:

1. A person who has never seen the booth succeeds at one jutsu within 20 seconds, unaided.
2. The system runs 4 hours continuously without a restart.
3. End-to-end latency stays under 100ms and frame rate above 30fps for the full session.
4. Held-out-person classification accuracy exceeds 92% with the confusion matrix documented.
5. Fewer than one false activation per 5 minutes of idle footage.
6. Three jutsu are fully polished with anchoring, audio and release, and the operator override works for all of them.
7. A full dry run has been completed on the real hardware in a comparable space.
8. The educational boards are printed and the operator script is written.
