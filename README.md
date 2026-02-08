# Red Object Tracking with Kalman Filtering

This project is a from-scratch computer vision tracker written in Python using
OpenCV and NumPy.

The goal is to reliably track red objects (currently a red marker cap) using a
Kalman filter with position, velocity, and acceleration. The system is designed
to be stable even when the object is stationary and to avoid creating duplicate
tracks from noise.

The code is intentionally written without OpenCV’s built-in tracking or Kalman
filter classes so the math and logic are fully visible and easy to modify.

---

## What This Does

- Detects red objects using HSV color filtering
- Tracks multiple objects at the same time
- Uses a constant-acceleration Kalman filter:
  - Position
  - Velocity
  - Acceleration
- Keeps object IDs stable over time
- Avoids false positives and flickering tracks
- Optionally draws object trajectories

Right now the detection is tuned for a red marker cap, but the structure is
ready to be extended to ping-pong balls or other objects.

---

## Requirements

- Python 3.8+
- OpenCV
- NumPy

Install dependencies with:

```bash
pip install opencv-python numpy
```
How to Run (For latest version replace main with prod)
```bash
python main.py
```

Press Q to quit.

Switching Between Camera and Video File

By default, the code uses your webcam:
```python
cap = cv2.VideoCapture(CAMERA_INDEX)
```

To use a video file instead, replace that line with:
```python
cap = cv2.VideoCapture("path/to/your_video.mp4")
```

Example:
```python
cap = cv2.VideoCapture("videos/test_clip.mp4")
```

No other code changes are needed.

Notes:

- Detection is blob-based rather than shape-based because marker caps are not
perfectly circular from most camera angles.

- The Kalman filter smooths motion and predicts object position when detections
are briefly lost.

- Track confirmation is used to prevent noise from creating fake objects.

Future Ideas: 

- Ping-pong ball tracking with circular detection

- Gravity and bounce physics

- Better data association (Hungarian algorithm)

- Real-world unit conversion using camera calibration
