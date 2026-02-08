import cv2
import numpy as np
import time
from itertools import count

# -------------------------
# CONFIG
# -------------------------
CAMERA_INDEX = 0

# Detection tuning (for marker cap)
MIN_CONTOUR_AREA = 200
ASPECT_RATIO_MIN = 0.3
ASPECT_RATIO_MAX = 3.0

# Tracking
MAX_ASSOCIATION_DIST = 120
MAX_MISSED_FRAMES = 25
MIN_CONFIRM_HITS = 3

# Kalman noise (low jitter for stationary objects)
PROCESS_NOISE = 0.2
MEASUREMENT_NOISE = 8.0

DRAW_TRAJECTORY = True
TRAJECTORY_LENGTH = 40

# Red HSV (for marker cap)
LOWER_RED_1 = np.array([0, 140, 120])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 140, 120])
UPPER_RED_2 = np.array([180, 255, 255])

KERNEL = np.ones((5, 5), np.uint8)
track_id_gen = count()

# -------------------------
# KALMAN FILTER (w/ ACCELERATION)
# -------------------------
class KalmanFilter2DAccel:
    """
    State: [x, y, vx, vy, ax, ay]
    """
    def __init__(self, initial_pos):
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.x[0:2, 0] = initial_pos

        self.P = np.eye(6, dtype=np.float32) * 400.0
        self.Q = np.eye(6, dtype=np.float32) * PROCESS_NOISE
        self.R = np.eye(2, dtype=np.float32) * MEASUREMENT_NOISE

        self.H = np.zeros((2, 6), dtype=np.float32)
        self.H[0, 0] = 1
        self.H[1, 1] = 1

    def predict(self, dt):
        dt2 = 0.5 * dt * dt

        F = np.array([
            [1, 0, dt, 0,  dt2, 0],
            [0, 1, 0,  dt, 0,  dt2],
            [0, 0, 1,  0,  dt, 0],
            [0, 0, 0,  1,  0,  dt],
            [0, 0, 0,  0,  1,  0],
            [0, 0, 0,  0,  0,  1],
        ], dtype=np.float32)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        z = z.reshape(2, 1)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def position(self):
        return self.x[0, 0], self.x[1, 0]

    def velocity(self):
        return self.x[2, 0], self.x[3, 0]

# -------------------------
# TRACK CLASS
# -------------------------
class Track:
    def __init__(self, pos):
        self.id = next(track_id_gen)
        self.kf = KalmanFilter2DAccel(pos)
        self.missed = 0
        self.hits = 1
        self.confirmed = False
        self.history = []

    def predict(self, dt):
        self.kf.predict(dt)
        if DRAW_TRAJECTORY:
            self.history.append(self.kf.position())
            self.history = self.history[-TRAJECTORY_LENGTH:]

    def update(self, z):
        self.kf.update(z)
        self.missed = 0
        self.hits += 1
        if self.hits >= MIN_CONFIRM_HITS:
            self.confirmed = True

# -------------------------
# DETECTION (BLOB-BASED)
# -------------------------
def detect_red_blobs(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h)

        if not (ASPECT_RATIO_MIN <= aspect <= ASPECT_RATIO_MAX):
            continue

        cx = x + w / 2
        cy = y + h / 2

        detections.append(np.array([cx, cy], dtype=np.float32))

    return detections, mask

# -------------------------
# DATA LINKING
# -------------------------
def associate_tracks(tracks, detections, dt):
    matches = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_detections = set(range(len(detections)))

    if not tracks or not detections:
        return matches, unmatched_tracks, unmatched_detections

    cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)

    for i, t in enumerate(tracks):
        px, py = t.kf.position()
        vx, vy = t.kf.velocity()

        pred_x = px + vx * dt
        pred_y = py + vy * dt

        for j, d in enumerate(detections):
            cost[i, j] = np.linalg.norm([pred_x - d[0], pred_y - d[1]])

    while True:
        i, j = np.unravel_index(np.argmin(cost), cost.shape)
        if cost[i, j] > MAX_ASSOCIATION_DIST:
            break

        matches.append((i, j))
        unmatched_tracks.discard(i)
        unmatched_detections.discard(j)

        cost[i, :] = np.inf
        cost[:, j] = np.inf

        if not np.isfinite(cost).any():
            break

    return matches, unmatched_tracks, unmatched_detections

# -------------------------
# THE MAIN LOOP
# -------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
tracks = []

last_time = time.time()
# WIP
while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    dt = max(now - last_time, 1e-3)
    last_time = now

    detections, mask = detect_red_blobs(frame)

    # Predict
    for t in tracks:
        t.predict(dt)

    matches, unmatched_tracks, unmatched_detections = associate_tracks(tracks, detections, dt)

    # Update matched tracks
    for ti, di in matches:
        tracks[ti].update(detections[di])

    # Age unmatched tracks
    for ti in unmatched_tracks:
        tracks[ti].missed += 1

    # Spawn new tracks
    for di in unmatched_detections:
        tracks.append(Track(detections[di]))

    # Remove dead tracks
    tracks = [t for t in tracks if t.missed < MAX_MISSED_FRAMES]

    # Draw confirmed tracks only
    for t in tracks:
        if not t.confirmed:
            continue

        x, y = t.kf.position()
        x, y = int(x), int(y)

        cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)
        cv2.putText(frame, f"ID {t.id}", (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if DRAW_TRAJECTORY and len(t.history) > 1:
            for i in range(1, len(t.history)):
                p1 = tuple(map(int, t.history[i - 1]))
                p2 = tuple(map(int, t.history[i]))
                cv2.line(frame, p1, p2, (0, 200, 255), 2)

    cv2.imshow("Stable Marker Cap Tracking", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
