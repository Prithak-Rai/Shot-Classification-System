# Padel Game Analytics - Shot Classification System

Computer Vision Project for analyzing padel gameplay video and classifying key shot types.

## Project Objective

Build a system that:

- Detects and tracks players, ball, and racket
- Classifies shot types (`Forehand`, `Backhand`, `Serve`)
- Exports predictions in structured formats (`JSON` and `CSV`)
- Produces a visualized output video with detection + shot overlay

## Features Implemented

- **Object Detection & Tracking**
  - Player tracking (YOLO tracking)
  - Ball detection per frame (YOLO)
  - Racket detection per frame (YOLO)
  - Racket-to-player association (YOLO)

- **Shot Classification**
  - Forehand
  - Backhand
  - Serve

- **Structured Output**
  - `output/shots.json`
  - `output/shots.csv`

### Bonus Tasks

- **Shot Analytics**
  - Total shot count
  - Shot-type counts
  - Per-player shot breakdown
  - Stored in `output/shots.json`

- **Visualization**
  - Bounding boxes for players, ball, and racket
  - Court line overlay
  - Shot event overlay text (type, player, timestamp)
  - Saved to `output/output.mp4`

## Repository Structure

- main.py - end-to-end pipeline entrypoint
- shot_classifier.py - shot detection/classification + output writing + analytics
- models/ - models to detect court, ball, racket
- trackers/ - player, ball, racket tracking modules
- court_line_detector/ - court keypoint detection and overlay
- utils/ - video I/O and geometry helper functions
- analysis/APPROACH.md - methodology/challenges/improvements write-up
- output/ - output video

## Setup

### 1) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Run

1. Put input video at `input/` 
2. Ensure model files exist:
   - `yolo12n.pt` (player model)
   - `models/best.pt` (ball model)
   - `models/racket/best.pt` (racket model)
   - `models/Court/best.pt` (court model)
3. Run:

```bash
python3 main.py
```

Generated outputs:

- `output/shots.json`
- `output/shots.csv`
- `output/output.mp4`