# 📷 Volleyball Tracker — Real-Time Ball Detection & Prediction

> Computer vision system for tracking and predicting volleyball trajectories using sensor-fused Kalman filtering.  
> Built in collaboration with a university professor as part of ongoing sports analytics research.

---

## What it does

Tracks a volleyball in real time from a standard webcam feed. It doesn't just detect where the ball *is* — it predicts where it's *going*, handles occlusion (when the ball disappears behind a player), and estimates depth using a single-camera pinhole model.

**Demo output:**
- Live tracked circle with ID + velocity arrow
- Fading trajectory trail
- Estimated Z-depth in meters
- FPS counter + detection count HUD
- Occlusion state indicator (`[OCC]`)

---

## How it works

### Three fused detectors

Rather than relying on a single detection method (which fails in changing lighting or fast motion), three independent detectors run per frame and their results are fused by weighted confidence score:

| Detector | Technique | Weight |
|---|---|---|
| **Color** | HSV masking for yellow/white volleyball | 40% |
| **Shape** | Hough circle transform + circularity validation | 35% |
| **Motion** | MOG2 background subtraction | 25% |

A detection only passes through if its fused confidence exceeds a threshold — this kills most false positives.

### 6D Kalman Filter

State vector: `[x, y, vx, vy, ax, ay]`

Most trackers use a 4D Kalman filter (position + velocity). This one tracks **acceleration** too, which matters for a ball in flight under gravity. The filter:
- Predicts position one frame ahead
- Corrects against fused measurements
- Coasts on prediction alone during occlusion (up to 20 frames)

### Greedy Hungarian-style association

Matches detections to existing tracks by minimizing distance cost, penalized by detection confidence. New unmatched detections spawn new tracks; unmatched tracks age out after 30 missed frames.

### Occlusion handling

When two confirmed tracks overlap spatially, both are flagged as occluded. The system trusts the Kalman prediction instead of the detector. When velocity vectors diverge enough, tracks are re-associated.

---

## Tech stack

```
Python 3.11+
OpenCV (cv2)     — capture, processing, Hough, MOG2, drawing
NumPy            — matrix math, Kalman equations
```

---

## Run it yourself

```bash
git clone https://github.com/blissio/Project-Camera-detection
cd Project-Camera-detection
pip install opencv-python numpy
python volleyball_tracker.py
```

Press `Q` to quit. Tested on a standard 1080p webcam at 30fps.

**Tune these constants in the config block if needed:**
- `CAMERA_INDEX` — which camera (0 = default)
- `FOCAL_LENGTH_PX` — calibrate to your camera for accurate Z-depth
- `FUSION_THRESHOLD` — raise if you're getting false detections

---

## Project status

Active development — research collaboration with Duquesne University.  
Upcoming: stereo camera support for true 3D depth, trajectory prediction for serve/spike classification.

---

*by [Rayane EL YASTI](https://github.com/blissio) · [blissio.github.io](https://blissio.github.io)*
