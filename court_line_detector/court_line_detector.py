from ultralytics import YOLO
import cv2

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, image):
        results = self.model(image)[0]

        keypoints = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            keypoints.append((int(cx), int(cy)))

        return keypoints

    def draw_court_lines(self, video_frames, keypoints):
        output_frames = []

        for frame in video_frames:
            for (x, y) in keypoints:
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            output_frames.append(frame)

        return output_frames
