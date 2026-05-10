# IMport All the Required Libraries
import os
import cv2
import pickle
from ultralytics import YOLO

class RacketTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True, conf=0.20)[0]
        racket_dict = {}
        for i, box in enumerate(results.boxes, start=1):
            if box.id is not None:
                track_id = int(box.id.tolist()[0])
            else:
                # Fallback for frames where tracker id is unavailable.
                track_id = i
            result = box.xyxy.tolist()[0]
            racket_dict[track_id] = result
        return racket_dict
    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        racket_detections = []
        if read_from_stub and stub_path is not None and os.path.isfile(stub_path):
            with open(stub_path, 'rb') as f:
                racket_detections = pickle.load(f)
                return racket_detections
        for frame in frames:
            racket_dict = self.detect_frame(frame)
            racket_detections.append(racket_dict)
        if stub_path is not None:
            d = os.path.dirname(stub_path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(racket_detections, f)
        return racket_detections
    
    def draw_bboxes(self, video_frames, racket_detections, player_detections=None, max_area_ratio=0.18):
        output_video_frames = []
        for idx, (frame, racket_dict) in enumerate(zip(video_frames, racket_detections)):
            player_dict = player_detections[idx] if player_detections is not None else {}
            for track_id, bbox in racket_dict.items():
                x1, y1, x2, y2 = bbox
                racket_area = max(1.0, (x2 - x1) * (y2 - y1))
                if track_id in player_dict:
                    px1, py1, px2, py2 = player_dict[track_id]
                    player_area = max(1.0, (px2 - px1) * (py2 - py1))
                    # Skip oversized racket boxes that overlap too much with player box.
                    if racket_area > max_area_ratio * player_area:
                        continue
                cv2.putText(frame, f'Racket ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames

    def associate_with_players(self, player_detections, racket_detections):
        """
        Assign at most one racket to each player per frame using nearest upper-body point.
        Returns list of dicts: {player_id: racket_bbox}.
        """
        associated = []
        prev_player_rackets = {}
        prev_player_miss = {}
        for player_dict, racket_dict in zip(player_detections, racket_detections):
            frame_assoc = {}
            used_rackets = set()
            for player_id, pb in player_dict.items():
                px1, py1, px2, py2 = pb
                # Approximate two hand zones from upper body corners.
                left_hand = (px1 + 0.22 * (px2 - px1), py1 + 0.35 * (py2 - py1))
                right_hand = (px2 - 0.22 * (px2 - px1), py1 + 0.35 * (py2 - py1))

                best_rid = None
                best_dist = float("inf")
                for rid, rb in racket_dict.items():
                    if rid in used_rackets:
                        continue
                    rx1, ry1, rx2, ry2 = rb
                    rc = ((rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0)
                    d_left = ((rc[0] - left_hand[0]) ** 2 + (rc[1] - left_hand[1]) ** 2) ** 0.5
                    d_right = ((rc[0] - right_hand[0]) ** 2 + (rc[1] - right_hand[1]) ** 2) ** 0.5
                    d = min(d_left, d_right)
                    if d < best_dist:
                        best_dist = d
                        best_rid = rid

                # Distance threshold scales with player bbox height.
                player_h = max(1.0, (py2 - py1))
                if best_rid is not None and best_dist < 0.9 * player_h:
                    frame_assoc[player_id] = racket_dict[best_rid]
                    used_rackets.add(best_rid)
                    prev_player_rackets[player_id] = racket_dict[best_rid]
                    prev_player_miss[player_id] = 0
                else:
                    # Keep last racket for a short gap to avoid flickering.
                    miss = prev_player_miss.get(player_id, 0) + 1
                    prev_player_miss[player_id] = miss
                    if player_id in prev_player_rackets and miss <= 4:
                        frame_assoc[player_id] = prev_player_rackets[player_id]
            associated.append(frame_assoc)
        return associated
