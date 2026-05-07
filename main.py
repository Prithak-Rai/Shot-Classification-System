import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")

video_path = "input/sample_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

frame_id = 0
frame_skip = 2
next_id = 0

# --- Persistent track memory ---
tracks = {}

MAX_MISSED_FRAMES = 10
MAX_DIST = 180

def get_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return interArea / float(areaA + areaB - interArea)

COLORS = [
    (0, 255, 0), (255, 100, 0), (0, 100, 255),
    (255, 0, 255), (0, 255, 255), (255, 255, 0)
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % frame_skip != 0:
        continue

    frame = cv2.resize(frame, (640, 480))
    results = model(frame, conf=0.25, imgsz=416, verbose=False)

    detections = []

    # ----------------------------
    # DETECTIONS
    # ----------------------------
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cx, cy = get_center(x1, y1, x2, y2)

            detections.append((cx, cy, x1, y1, x2, y2, conf))

    # ----------------------------
    # FIXED TRACKING LOGIC
    # ----------------------------
    matched_track_ids = set()
    matched_det_indices = set()

    track_ids = list(tracks.keys())

    for det_i, det in enumerate(detections):

        cx, cy, x1, y1, x2, y2, conf = det

        best_id = None
        best_score = -1

        for track_id in track_ids:

            if track_id in matched_track_ids:
                continue

            info = tracks[track_id]

            # IoU score
            iou_score = iou((x1, y1, x2, y2), info['bbox'])

            # Distance score
            tx, ty = info['center']
            dist = np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
            dist_score = max(0, 1 - dist / MAX_DIST)

            score = 0.75 * iou_score + 0.25 * dist_score

            if score > best_score:
                best_score = score
                best_id = track_id

        # ----------------------------
        # MATCH FOUND
        # ----------------------------
        if best_id is not None and best_score > 0.15:

            tracks[best_id] = {
                'center': (cx, cy),
                'bbox': (x1, y1, x2, y2),
                'missed': 0
            }

            matched_track_ids.add(best_id)
            matched_det_indices.add(det_i)

        # ----------------------------
        # NEW TRACK
        # ----------------------------
        else:
            tracks[next_id] = {
                'center': (cx, cy),
                'bbox': (x1, y1, x2, y2),
                'missed': 0
            }
            next_id += 1

    # ----------------------------
    # AGE OUT TRACKS
    # ----------------------------
    dead_ids = []

    for track_id in tracks:
        if track_id not in matched_track_ids:
            tracks[track_id]['missed'] += 1

            if tracks[track_id]['missed'] > MAX_MISSED_FRAMES:
                dead_ids.append(track_id)

    for tid in dead_ids:
        del tracks[tid]

    # ----------------------------
    # DRAW
    # ----------------------------
    for track_id, info in tracks.items():

        if info['missed'] > 0:
            continue

        x1, y1, x2, y2 = info['bbox']
        color = COLORS[track_id % len(COLORS)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Player {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()