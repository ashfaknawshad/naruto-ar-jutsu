export interface CameraSession {
  stream: MediaStream;
  deviceId: string;
}

/**
 * Requests camera permission (so device labels are populated), lists video
 * inputs, and opens a stream for the given (or first) device.
 */
export async function listVideoDevices(): Promise<MediaDeviceInfo[]> {
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((d) => d.kind === "videoinput");
}

export async function openCamera(deviceId?: string): Promise<CameraSession> {
  const constraints: MediaStreamConstraints = {
    audio: false,
    video: deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
  };

  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  const track = stream.getVideoTracks()[0];
  const settings = track.getSettings();

  // Exposure/white-balance locking where the platform supports it. Silently
  // no-ops on browsers that don't expose these constraints (most phones).
  const capabilities = track.getCapabilities?.() as MediaTrackCapabilities & {
    exposureMode?: string[];
    whiteBalanceMode?: string[];
  };
  if (capabilities?.exposureMode?.includes("manual")) {
    try {
      await track.applyConstraints({ advanced: [{ exposureMode: "manual" } as never] });
    } catch {
      // Not fatal — fall back to auto-exposure.
    }
  }
  if (capabilities?.whiteBalanceMode?.includes("manual")) {
    try {
      await track.applyConstraints({ advanced: [{ whiteBalanceMode: "manual" } as never] });
    } catch {
      // Not fatal — fall back to auto white balance.
    }
  }

  return { stream, deviceId: settings.deviceId ?? deviceId ?? "" };
}

/**
 * Wires a <select> element to the available cameras and switches the given
 * <video> element's stream whenever the selection changes.
 */
export async function initDevicePicker(
  select: HTMLSelectElement,
  video: HTMLVideoElement,
  onSwitch: (session: CameraSession) => void,
): Promise<CameraSession> {
  const initial = await openCamera();
  video.srcObject = initial.stream;
  await video.play();

  const devices = await listVideoDevices();
  select.innerHTML = "";
  for (const d of devices) {
    const opt = document.createElement("option");
    opt.value = d.deviceId;
    opt.textContent = d.label || `Camera ${select.length + 1}`;
    if (d.deviceId === initial.deviceId) opt.selected = true;
    select.appendChild(opt);
  }

  select.addEventListener("change", async () => {
    for (const track of video.srcObject instanceof MediaStream ? video.srcObject.getTracks() : []) {
      track.stop();
    }
    const session = await openCamera(select.value);
    video.srcObject = session.stream;
    await video.play();
    onSwitch(session);
  });

  return initial;
}
