# Gesture Recognition

Real-time webcam hand-gesture recognition. [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python) finds 21 hand landmarks; OpenCV captures the camera and draws the overlay; a small **rule-based** classifier maps those landmarks to a few everyday poses.

No custom neural-net weights are trained or committed. On first run the official MediaPipe `hand_landmarker.task` bundle (~7.5 MB) is downloaded into `~/.cache/gesture-recognition/`.

## Gestures supported

| On-screen label | Pose |
| --- | --- |
| Open Palm | Index, middle, ring, and pinky extended (thumb optional) |
| Fist | Those four fingers folded; thumb not pointing up |
| Thumbs Up | Thumb extended and pointing up; other fingers folded |
| Victory | Index and middle extended; ring and pinky folded (peace sign) |
| Pointing | Index extended; middle, ring, and pinky folded |
| Unknown | A hand is visible but the pose does not match a rule |
| No Hand | No hand in view |

Hold the palm roughly toward the camera. Lighting and a clear view of the fingers help.

### How the rules work

MediaPipe returns 21 normalized landmarks (wrist, thumb joints, and four fingers with MCP / PIP / DIP / tip). The classifier never looks at pixels — only at those points.

- **Palm width** is the distance between the index and pinky MCP joints. Thresholds are fractions of that width so the rules scale with hand size and camera distance.
- A **non-thumb finger is extended** when the tip is clearly farther from its MCP than the PIP is, and the MCP-to-tip length is at least ~0.55× palm width. Folded fingers collapse the tip back toward the knuckle.
- The **thumb is extended** when its tip is away from the index MCP (not tucked across the palm) and farther from the wrist than the IP joint. **Thumbs up** additionally requires the tip to sit above the thumb MCP and wrist (image *y* grows downward).
- Labels are checked from **most specific to least**: Victory → Pointing → Thumbs Up → Open Palm → Fist. That way a peace sign is not reported as an open palm.

This is a v1 heuristic. Side-on views, extreme foreshortening, or fingers bunched together can fall through to **Unknown**.

## Setup

Python 3.10+ (3.12 is a good default). A webcam is required for the live demo.

```bash
git clone https://github.com/raptortreats/Gesture-Recognition.git
cd Gesture-Recognition

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional, so `python -m gesture_recognition` works without setting `PYTHONPATH`:

```bash
pip install -e .
```

The first launch downloads the Hand Landmarker model (needs network once). To use a file you already have:

```bash
python main.py --model-path /path/to/hand_landmarker.task
```

## Run

From the repo root (uses `main.py`, no package install required):

```bash
python main.py
```

Or, after `pip install -e .`:

```bash
python -m gesture_recognition
# or:  gesture-recognition
```

A window opens with the mirrored webcam feed, landmark overlay, the current gesture, and FPS. **Press `q`** (or Esc) to quit.

### Useful flags

| Flag | Meaning |
| --- | --- |
| `--camera 1` | Use camera index 1 instead of 0 |
| `--no-mirror` | Do not flip the preview (default is selfie-style) |
| `--max-hands 1` | Detect at most one hand |
| `--min-detection-confidence 0.6` | Stricter palm detection (0–1) |
| `--min-presence-confidence 0.6` | Stricter hand-presence score |
| `--min-tracking-confidence 0.6` | Stricter landmark tracking |
| `--width 1280 --height 720` | Request a capture size |
| `--image photo.jpg` | Classify a still image instead of the webcam |
| `--output out.jpg` | Write the annotated frame to disk |
| `--no-window` | Skip the OpenCV window (headless; pair with `--output` or `--image`) |
| `--model-path FILE` | Use a local `hand_landmarker.task` |

Example — annotate a photo without a display:

```bash
python main.py --image photo.jpg --output annotated.jpg --no-window --no-mirror
```

If the camera cannot be opened or frames stop arriving, the app prints a short error and exits with status 1 instead of crashing.

## Project layout

```
main.py                         # `python main.py` entrypoint
requirements.txt                # pinned OpenCV, MediaPipe, NumPy
src/gesture_recognition/
  cli.py                        # argparse
  config.py                     # defaults and model URL
  camera.py                     # webcam open / read failures
  model.py                      # first-run model download
  landmarks.py                  # MediaPipe Hand Landmarker wrapper
  classifier.py                 # rule-based gestures
  overlay.py                    # skeleton, label, FPS
  app.py                        # camera / image loop
tests/                          # classifier + CLI unit tests
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Classifier tests use synthetic landmarks (no webcam, no model file).

## License

This project's application code is MIT-licensed (see `LICENSE`). MediaPipe and the Hand Landmarker bundle are provided by Google under their own licenses (Apache 2.0 for MediaPipe).
