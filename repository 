# Edge Driver Safety System

**Repository template for an intelligent, edge-deployed driver safety system (SLM on-vehicle).**

---

## Project overview

Modern in-car alerts are limited and often cloud-dependent. This repo provides a complete starter GitHub repository structure, documentation, and example code to build an **onboard Small Language Model (SLM) agent** that detects driver fatigue, stress, and sudden unconsciousness in real time and performs adaptive, privacy-preserving safety actions on the vehicle.

Key goals:

* Real-time perception (camera + physiological) running on vehicle CPU/GPU
* On-device SLM agent for conversational, adaptive, empathic interactions
* Safety-first actions (alerts, controlled deceleration, hazard lights, call SOS)
* Privacy-preserving: no raw personal data leaves the vehicle
* Modular: swap perception models, sensors, and action policies

---

## Highlights / Features

* Face/eye/pose based fatigue detection (blink rate, PERCLOS, gaze)
* Stress detection from physiological signals (heart-rate variability) + facial micro-expressions
* Sudden unconsciousness detection using multi-modal fusion (loss of posture, prolonged eye closure, heart-rate drop, steering inactivity)
* On-device SLM (quantized + distilled) for verbal interaction, logging, and decision rationale
* Policy engine for graded actions (gentle alert → conversation → vehicle control fallback)
* Secure local storage & differential privacy features for optional telemetry
* ROS2-compatible nodes and an optional standalone Python service

---

## Suggested hardware & software

**Hardware (examples):**

* Edge platform: NVIDIA Jetson Orin/NX, Jetson Xavier; alternatively Intel NUC with GPU; or Automotive-grade SoC
* Camera: IR-capable NIR / RGB camera (for low-light)
* Optional sensors: PPG/heart-rate sensor, steering torque sensor, accelerometer

**Software stack:**

* Ubuntu 22.04 or Yocto-based embedded OS
* ROS 2 Humble or later (optional)
* Python 3.10+
* PyTorch (for training/ONNX export) or TensorRT / OpenVINO for inference
* ONNX Runtime Mobile, bitsandbytes + quantized SLM runtime

---

## Architecture (short)

1. **Perception**: Camera + sensors → light-weight CV models (face detection, landmarking, blink/gaze estimation). Runs at 10–30 FPS depending on hardware.
2. **Signal processing & feature extraction**: Compute blink rate, PERCLOS, gaze deviation, HRV metrics.
3. **Multi-modal fusion**: Rule + learned fusion model to compute `driver_state` probabilities: {alert, fatigued, stressed, unconscious}
4. **Policy & action manager**: Maps `driver_state` + context (speed, traffic) → actions (vibro seat, voice, haptic steering, lane-keep assist, safe-stop)
5. **SLM agent**: Small quantized LLM runs locally to converse, empathize, ask questions, and explain decisions.
6. **Privacy & logging**: Only store anonymized features and short transcripts locally; optional encrypted telemetry with explicit consent.

---

## Folder structure (template)

```
edge-driver-safety-system/
├─ README.md
├─ LICENSE
├─ .github/
│  ├─ workflows/          # CI (lint, tests, build artifacts)
│  └─ ISSUE_TEMPLATE.md
├─ docs/
│  ├─ architecture.md
│  ├─ privacy.md
│  └─ deploy_jetson.md
├─ modules/
│  ├─ perception/         # camera & CV models, inference wrappers
│  │  ├─ model/           # ONNX / TensorRT model artifacts (gitignored large files)
│  │  ├─ src/
│  │  │  ├─ detector.py
│  │  │  └─ landmarks.py
│  │  └─ tests/
│  ├─ signals/            # physiological sensor processing (PPG)
│  ├─ fusion/             # multi-modal fusion models
│  ├─ policy/             # policy engine to decide actions
│  └─ slm_agent/          # local SLM runtime & conversation planner
├─ ros_nodes/             # ROS2 node wrappers (optional)
├─ deploy/                # Dockerfiles, systemd services
├─ examples/              # demo scripts to run on laptop or edge device
├─ data/                  # dataset pointers & preprocessing scripts (no raw data)
└─ tools/                 # utilities: calibration, benchmarks, dataset downloaders
```

---

## Example README sections created for each module (included in repo)

* `modules/perception/README.md` — quick-run instructions, expected camera frames, calibration
* `modules/slm_agent/README.md` — how to load quantized SLM, quantization tips, latency targets
* `docs/privacy.md` — storage rules, consent channels, optional remote upload with end-to-end encryption

---

## Starter files included (samples)

This template repo contains *starter code snippets* and runnable demos:

* `examples/local_demo.py` — run simplified pipeline on a laptop with webcam
* `modules/perception/src/detector.py` — lightweight face detector wrapper using ONNXRuntime
* `modules/fusion/simple_fusion.py` — rule-based fusion to combine blink/perclos/HRV
* `modules/slm_agent/agent.py` — example wrapper to load a quantized SLM and generate response given a driver state
* `ros_nodes/driver_monitor_node.py` — ROS2 node that publishes `DriverState` and subscribes to `vehicle/status`

(Models and large artifacts are .gitignored; instructions to download prebuilt models are in `docs/`.)

---

## Quick start (developer laptop demo)

1. Create virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the demo with your webcam:

```bash
python examples/local_demo.py --camera 0
```

3. Observe console logs and the simple web UI at `http://localhost:8080` (optional) showing driver state.

---

## Safety & fail-safe behaviour

* The system is **advisory** by default. Any vehicle control actions must be gated by the vehicle's safety controller and certified safe policies.
* Include hardware and software watchdogs; if SLM or perception fails, fall back to the vehicle default alert system.
* Keep conservative thresholds at higher speeds; escalate gracefully.
* Thorough testing required before any real-vehicle deployment — see `docs/safety_testing.md`.

---

## Privacy

* Raw camera frames should not be logged persistently. Store only derived features (blink-rate, PERCLOS), short hashed IDs and consented transcripts.
* Provide user toggle to disable SLM logging. Implement local-only mode by default.

---

## Model & data guidance

* Use small, distilled models for on-device SLM (e.g., 1–3B parameter distilled versions quantized to 4/8-bit) — libraries: `llama.cpp`, `GGML`, `bitsandbytes` for quantized weights.
* Perception models: MobileNetV3 / BlazeFace / MediaPipe style detectors exported to ONNX.
* Datasets (examples): Drowsiness datasets (NTHU-DDD), eye-blink datasets, stress detection datasets (WESAD) — include download scripts but do not include raw datasets in repo.

---

## CI / Testing

* Unit tests for signal-processing functions
* Integration test: run short synthetic video and check that `DriverState` messages publish in expected ranges
* Performance tests to ensure inference latency < target (e.g., 50–100ms per frame on target hardware)

---

## Roadmap & milestones

* `v0.1` — basic perception + fusion + demo SLM answer
* `v0.2` — ROS2 integration + hardware-in-the-loop test
* `v1.0` — safety-audited policy engine + controlled stop capability

---

## Contributing

Please read `CONTRIBUTING.md` for code style, commit messages, and how to submit PRs. All changes to safety-critical policy modules must include tests and a risk assessment document.

---

## License

This template is released under the **MIT License** (see `LICENSE`).

---

## Next steps I can help with

* Generate the actual `examples/local_demo.py` runable script
* Create the ROS2 `driver_monitor_node.py` with message definitions
* Produce a quantized SLM loading wrapper (example for a 1B-3B distilled model)
* Build GitHub Actions workflow for tests and cross-platform builds

---

*End of template.*

---

# Python minimal runnable code

Below are the files added to the repo for a Python-only demo. Save each code block into the path shown and then run the demo as described in `README.md`.

---

## File: requirements.txt

```
opencv-python
mediapipe
numpy
flask
```

---

## File: README_PYTHON.md

````
# Python demo (local_demo)

## Setup

1. Create virtual environment and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

2. Run the demo (webcam required):

```bash
python examples/local_demo.py --camera 0
```

3. The demo will open a simple OpenCV window showing camera feed and driver state printed on top. The SLM agent will print conversational responses to the console.

```

---

## File: examples/local_demo.py

```

"""
Simple local demo: webcam -> MediaPipe face mesh -> blink & PERCLOS approximations -> fusion -> SLM agent responses

This is a lightweight demo and uses heuristic thresholds for illustration only.
"""
import time
import argparse
import cv2
import numpy as np
import mediapipe as mp
from modules.fusion.simple_fusion import estimate_driver_state
from modules.slm_agent.agent import SLMAgent

# Mediapipe setup

mp_face_mesh = mp.solutions.face_mesh

# Eye landmark indices from Mediapipe face mesh (right/left)

# Using averaged eye aspect from several landmarks

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, idx_list, image_w, image_h):
# Convert normalized landmarks to pixel coords
pts = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in idx_list]
# vertical distances
A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
# horizontal distance
C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
if C == 0:
return 0.0
ear = (A + B) / (2.0 * C)
return ear

def main(camera_idx=0):
cap = cv2.VideoCapture(camera_idx)
if not cap.isOpened():
print('ERROR: cannot open camera')
return

```
slm = SLMAgent()

# stateful counters
blink_count = 0
frame_count = 0
closed_frames = 0
perclos_window = 150  # number of frames for PERCLOS (~5-10s depending on fps)
perclos_history = []

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w = image.shape[:2]

        ear = None
        eye_closed = False

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            left_ear = eye_aspect_ratio(lm, LEFT_EYE_IDX, w, h)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE_IDX, w, h)
            ear = (left_ear + right_ear) / 2.0

            # heuristic thresholds (tune per-device)
            if ear < 0.20:
                closed_frames += 1
                eye_closed = True
            else:
                if closed_frames > 2 and closed_frames < 15:
                    blink_count += 1
                closed_frames = 0

        # update PERCLOS history (1 if eyes closed, 0 otherwise)
        perclos_history.append(1 if eye_closed else 0)
        if len(perclos_history) > perclos_window:
            perclos_history.pop(0)
        perclos = sum(perclos_history) / max(1, len(perclos_history))

        # Simulated HRV / physiological signal (demo only)
        # In a real system you'd read from a PPG or CAN bus sensor
        simulated_hrv = 50.0  # placeholder value (ms)

        # Estimate driver state using fusion
        driver_state = estimate_driver_state(blink_count=blink_count, perclos=perclos, hrv=simulated_hrv)

        # Ask SLM agent to produce a response when state is non-alert and when state changed
        if driver_state != 'alert' and frame_count % 40 == 0:
            response = slm.respond(driver_state)
            print(f"SLM: {response}")

        # overlay information
        cv2.putText(image, f"State: {driver_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if ear is not None:
            cv2.putText(image, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        cv2.putText(image, f"PERCLOS: {perclos:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

        cv2.imshow('Driver Monitor Demo', image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
```

if **name** == '**main**':
parser = argparse.ArgumentParser()
parser.add_argument('--camera', type=int, default=0)
args = parser.parse_args()
main(camera_idx=args.camera)

```

---

## File: modules/fusion/simple_fusion.py

```

"""
Simple rule-based fusion for demo purposes.
Takes blink_count, perclos (0..1), hrv (ms) and returns one of:
'alert', 'fatigued', 'stressed', 'unconscious'

This is illustrative only — replace with trained models for production.
"""

def estimate_driver_state(blink_count: int, perclos: float, hrv: float) -> str:
# Heuristic thresholds (tunable)
if perclos > 0.6:
return 'unconscious'
if perclos > 0.25 or blink_count > 20:
return 'fatigued'
# low HRV value may indicate stress (lower HRV => stressed)
if hrv < 30:
return 'stressed'
return 'alert'

```

---

## File: modules/slm_agent/agent.py

```

"""
A tiny simulated SLM agent that responds empathetically.
In production you'd load a quantized LLM (llama.cpp / ggml / onnx runtime) and run it on-device.
"""

from typing import Dict

class SLMAgent:
def **init**(self):
# small rule-based prompt-response for demo
self.templates: Dict[str, str] = {
'alert': "You're looking good — keep focus. Let me know if you need a break.",
'fatigued': "I detect signs of drowsiness. Please take a short break or pull over when safe. Would you like me to set a gentle alarm?",
'stressed': "You seem stressed. Try a few deep breaths. I can lower cabin notifications and give you a calming voice prompt.",
'unconscious': "Warning: driver unresponsive. Initiating safety measures: turning on hazard lights and slowing vehicle. Contacting emergency services if needed.",
}

```
def respond(self, state: str) -> str:
    # In a real SLM you'd condition on recent context and generate token-by-token.
    return self.templates.get(state, "I'm here to help.")
```

```

---

## Notes

- These files are intended as a minimal, runnable demonstration. They avoid heavy dependencies like `dlib` and use `mediapipe` for facial landmarks.
- Replace simulated HRV and thresholds with real sensor input and a trained fusion model for real deployments.

---

_End of added Python demo files._
```
