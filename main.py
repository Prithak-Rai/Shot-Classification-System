from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker

def main():
    #Read the input video
    input_video_path = 'input/a.mp4'
    video_frames = read_video(input_video_path)

    # Detect and track Players
    player_tracker = PlayerTracker(model_path='yolo12n.pt')
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")

    #Detect Ball
    ball_tracker = BallTracker(model_path='models/best.pt')
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")

    #Draw output
    #Draw Players bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    #Draw Ball bounding boxes
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    #Save Video
    save_video(output_video_frames, "output/output.mp4")

if __name__ == "__main__":
    main()