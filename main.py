import cv2
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker, RacketTracker
from court_line_detector import CourtLineDetector
from shot_classifier import classify_shots, compute_shot_analytics, save_shot_outputs


# ------------------------------------------------------------------ #
#  SHOT OVERLAY                                                        #
# ------------------------------------------------------------------ #
def overlay_shot_events(frames, shots, fps):
    """
    Burns the most recent shot label onto each frame for 1.2 s.
    Also shows the evidence score so you can tune the threshold visually.
    """
    shots_by_frame  = {s["frame"]: s for s in shots}
    hold_frames     = int(1.2 * fps)
    last_shot       = None
    last_shot_frame = -(10 ** 9)

    for idx, frame in enumerate(frames):
        if idx in shots_by_frame:
            last_shot       = shots_by_frame[idx]
            last_shot_frame = idx

        if last_shot is None:
            continue
        if idx - last_shot_frame > hold_frames:
            continue

        ev   = last_shot.get("evidence", 0.0)
        text = (
            f"Shot: {last_shot['shot_type']}  |  "
            f"Player: {last_shot['player_id']}  |  "
            f"t={last_shot['timestamp_sec']:.2f}s  |  "
            f"ev={ev:.2f}"
        )
        cv2.rectangle(frame, (20, 20), (920, 70), (0, 0, 0), -1)
        cv2.putText(
            frame, text,
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (0, 255, 255),
            2,
        )


# ------------------------------------------------------------------ #
#  MAIN                                                                #
# ------------------------------------------------------------------ #
def main():

    input_video_path = "input/sample_video.mp4"
    video_frames     = read_video(input_video_path)

    # ── Player tracking ──────────────────────────────────────────────
    player_tracker    = PlayerTracker(model_path="yolo12n.pt")
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/player.pkl",
    )

    # ── Ball tracking ────────────────────────────────────────────────
    ball_tracker    = BallTracker(model_path="models/best.pt")
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/ball.pkl",
    )

    # ── Racket tracking ──────────────────────────────────────────────
    racket_tracker    = RacketTracker(model_path="models/racket/best.pt")
    racket_detections = racket_tracker.detect_frames(
        video_frames,
        read_from_stub=False,
        stub_path="tracker_stubs/racket.pkl",
    )
    # associate_with_players returns a list[dict] {player_id: racket_bbox}
    # which is the format classify_shots expects for racket_detections.
    racket_detections = racket_tracker.associate_with_players(
        player_detections,
        racket_detections,
    )

    # ── Court detection ──────────────────────────────────────────────
    court_detector = CourtLineDetector("models/Court/best.pt")
    court_keypoints = [court_detector.predict(f) for f in video_frames]

    # ── Shot classification ──────────────────────────────────────────
    fps = 30

    shots = classify_shots(
        ball_detections,
        player_detections,
        fps,
        racket_detections=racket_detections,   # now wired in as soft bonus
        evidence_threshold=0.40,               # tune: raise → fewer but cleaner hits
        smooth_window=5,                       # tune: raise → smoother trajectory
    )

    analytics = compute_shot_analytics(shots)
    save_shot_outputs("sample_video.mp4", shots, analytics, output_dir="output")

    print("[INFO] Shot analytics:")
    print(f"       Total shots : {analytics['total_shots']}")
    print(f"       Breakdown   : {analytics['shot_counts']}")
    print(f"       Avg evidence: {analytics['avg_evidence']}")

    # ── Draw everything ──────────────────────────────────────────────
    output_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_frames = ball_tracker.draw_bboxes(output_frames, ball_detections)
    output_frames = racket_tracker.draw_bboxes(
        output_frames,
        racket_detections,
        player_detections=player_detections,
        max_area_ratio=0.18,
    )

    for i in range(len(output_frames)):
        output_frames[i] = court_detector.draw_court_lines(
            [output_frames[i]],
            court_keypoints[i],
        )[0]

    overlay_shot_events(output_frames, shots, fps)
    save_video(output_frames, "output/output.mp4")
    print("[INFO] Done — output/output.mp4")


if __name__ == "__main__":
    main()