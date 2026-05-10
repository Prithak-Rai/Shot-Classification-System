# Approach Explanation - Padel Game Analytics

## 1) Methodology

This project uses a CV pipeline to process a padel match video and classify shots.

Pipeline :

1. Read input video as frames (`utils/video_utils.py`)
2. Detect and track players (`trackers/player_tracker.py`)
3. Detect ball per frame (`trackers/ball_tracker.py`)
4. Detect rackets and associate each racket to nearest player (`trackers/racket_tracker.py`)
5. Detect court lines/keypoints (`court_line_detector/court_line_detector.py`)
6. Classify shots from ball motion (`shot_classifier.py`)
7. Export structured outputs (`output/shots.json`, `output/shots.csv`)
8. Output video (`output/output.mp4`)

## 2) Shot Classification Logic

- Ball center from detection boxes
- Velocity and speed calculation
- Candidate hit detection using:
  - acceleration or
  - direction reversal of the ball around player's frame
- Closest-player shot identified using left/right hand position
- Shot type classification:
  - Forehand / Backhand based on side of contact relative to player center and hand side
  - Serve based on upper-body contact with fast speed 

## 3) Structured Outputs

Per-shot output fields:

- `frame`
- `timestamp_sec`
- `shot_type`
- `player_id`

Analytics output includes:

- `total_shots`
- `shot_counts` i.e; forehand/backhand/serve totals
- `player_breakdown` i.r; shot count per player

## 4) Challenges Faced

1. **Ball detection noise and misses**
   - Small fast-moving ball causes the model to not detect the ball properly 
   - Mitigation: use motion-based hit rule

2. **Player-racket **
   - Racket detector IDs are not stable
   - Mitigation: associated each racket to nearest player

3. **Noisy shot boundaries**
   - A single swing creating multiple shot events
   - Mitigation: cooldown after each analyzed hit

## 5) Improvements for Next Iteration

1. Track the whole player body's movement and determine the shot.
2. Collect padel racekt data/Train the model with proper data to make the racket detection more accurate.
3. Crop the video with precision to make the noisey video a bit proper to analyze.
4. Train a sequence model (LSTM)  for shot-type classification.
5. Add a small dashboard for full match analysis.
