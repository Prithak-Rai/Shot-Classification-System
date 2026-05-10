import os
import csv
import json
from collections import Counter, defaultdict
from utils import get_center_bbox, measure_distance


PADEL_MIN_SPEED_PX   = 1.8   
PADEL_FAST_SPEED_PX  = 7.0   
PADEL_SMASH_SPEED_PX = 5.0   
PADEL_COOLDOWN_SEC   = 0.15  


# Save output
def save_shot_outputs(video_name, shots, analytics=None, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "shots.json")
    with open(json_path, "w") as f:
        json.dump({"video": video_name, "shots": shots, "analytics": analytics}, f, indent=2)

    csv_path = os.path.join(output_dir, "shots.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame", "timestamp_sec", "shot_type", "player_id", "evidence"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for s in shots:
            writer.writerow(s)

    print(f"[INFO] Saved {len(shots)} shots → {output_dir}/shots.json + shots.csv")


# Ball center
def ball_center(ball_dict):
    if not ball_dict:
        return None
    bbox = next(iter(ball_dict.values()))
    return get_center_bbox(bbox)


# Smoothen the trajectory
def _median_list(lst):
    s   = sorted(lst)
    n   = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else 0.5 * (s[mid - 1] + s[mid])


def smooth_positions(ball_positions, window=5):
    """
    Median-smooth x and y independently over `window` frames.
    Kills single-frame spikes without smearing genuine direction changes.
    """
    n    = len(ball_positions)
    half = window // 2
    out  = []
    for i in range(n):
        lo  = max(0, i - half)
        hi  = min(n, i + half + 1)
        pts = [ball_positions[k] for k in range(lo, hi) if ball_positions[k] is not None]
        if pts:
            out.append((_median_list([p[0] for p in pts]),
                        _median_list([p[1] for p in pts])))
        else:
            out.append(None)
    return out


def interpolate_missing(ball_positions, max_gap=8):
    pos = list(ball_positions)
    n   = len(pos)
    i   = 0
    while i < n:
        if pos[i] is None:
            gap_start = i - 1
            while gap_start >= 0 and pos[gap_start] is None:
                gap_start -= 1
            if gap_start < 0:
                j = i
                while j < n and pos[j] is None:
                    j += 1
                i = j + 1
                continue
            j = i
            while j < n and pos[j] is None:
                j += 1
            gap_end = j
            gap_len = gap_end - gap_start - 1
            if gap_end < n and gap_len <= max_gap:
                p0 = pos[gap_start]
                p1 = pos[gap_end]
                for k in range(1, gap_len + 1):
                    t = k / (gap_len + 1)
                    pos[gap_start + k] = (
                        p0[0] + t * (p1[0] - p0[0]),
                        p0[1] + t * (p1[1] - p0[1]),
                    )
            i = j + 1
        else:
            i += 1
    return pos

# Speed / veloity helpers
def speed(v):
    if v is None:
        return 0.0
    return (v[0] ** 2 + v[1] ** 2) ** 0.5


def x_direction_changed(v1, v2, min_turn_px=0.8):
    if v1 is None or v2 is None:
        return False
    vx1, vx2 = v1[0], v2[0]
    return (vx1 * vx2) < 0 and (abs(vx1) + abs(vx2)) >= min_turn_px


def y_direction_changed(v1, v2, min_turn_px=0.8):
    if v1 is None or v2 is None:
        return False
    vy1, vy2 = v1[1], v2[1]
    return (vy1 * vy2) < 0 and (abs(vy1) + abs(vy2)) >= min_turn_px


def direction_changed(v1, v2):
    if v1 is None or v2 is None:
        return False
    return (v1[0] * v2[0] + v1[1] * v2[1]) < 0


def deflection_cosine(v1, v2):
    """
    Cosine of angle between incoming and outgoing velocity.
    -1 = perfect reversal, 0 = 90° deflection, +1 = no change.
    """
    s1, s2 = speed(v1), speed(v2)
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0
    return (v1[0] * v2[0] + v1[1] * v2[1]) / (s1 * s2)


def point_to_bbox_distance(point, bbox):
    px, py = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return (dx * dx + dy * dy) ** 0.5


# Determine the player side i.e; top/bottom for shot identification
def compute_court_midline(player_detections_sequence):
    """
    Compute a STABLE court midline by averaging player y-positions
    over all frames. This is more reliable than per-frame median.
    Returns the y value that divides top players from bottom players.
    """
    all_cys = []
    for players in player_detections_sequence:
        for bbox in players.values():
            cy = (bbox[1] + bbox[3]) / 2.0
            all_cys.append(cy)
    if not all_cys:
        return None
    all_cys.sort()
    #midline
    n = len(all_cys)
    return all_cys[n // 2]


def player_side(players, pid, court_midline=None):

    if pid not in players:
        return "unknown"

    target_cy = (players[pid][1] + players[pid][3]) / 2.0

    if court_midline is not None:
        return "top" if target_cy < court_midline else "bottom"

    # per-frame median
    all_cys = sorted((b[1] + b[3]) / 2.0 for b in players.values())
    n   = len(all_cys)
    mid = n // 2
    midline = all_cys[mid] if n % 2 == 1 else (all_cys[mid-1] + all_cys[mid]) / 2.0
    return "top" if target_cy < midline else "bottom"


#Scoring
def compute_evidence(v_in, v_out, p_cur, player_bbox, dist_to_player, racket_frame):
    ev = {}

    # Racket proximity
    racket_score = 0.0
    if racket_frame:
        bx, by = p_cur
        for rbbox in racket_frame.values():
            rx1, ry1, rx2, ry2 = rbbox
            rw = max(1.0, rx2 - rx1)
            rh = max(1.0, ry2 - ry1)
            ex1 = rx1 - 2.0 * rw;  ey1 = ry1 - 2.0 * rh
            ex2 = rx2 + 2.0 * rw;  ey2 = ry2 + 2.0 * rh
            if ex1 <= bx <= ex2 and ey1 <= by <= ey2:
                racket_score = 0.20
                break
    ev["racket"] = racket_score

    max_spd  = max(speed(v_in), speed(v_out))
    spd_norm = min(1.0, max(0.0,
        (max_spd - PADEL_MIN_SPEED_PX) / (PADEL_FAST_SPEED_PX - PADEL_MIN_SPEED_PX)
    ))
    ev["speed"] = round(0.30 * spd_norm, 4)

    cos_a        = deflection_cosine(v_in, v_out)
    deflect_norm = (1.0 - cos_a) / 2.0   # -1→1.0, 0→0.5, +1→0.0
    ev["deflect"] = round(0.35 * deflect_norm, 4)

    player_h  = max(1.0, player_bbox[3] - player_bbox[1])
    prox_norm = max(0.0, 1.0 - dist_to_player / (3.5 * player_h))
    ev["prox"] = round(0.15 * prox_norm, 4)

    ev["total"] = round(ev["racket"] + ev["speed"] + ev["deflect"] + ev["prox"], 4)
    return ev


# Shot classification
def classify_shot_type(p_cur, v_in, v_out, bbox, p_side):

    x1, y1, x2, y2 = bbox
    player_w  = max(1.0, x2 - x1)
    player_h  = max(1.0, y2 - y1)
    px_center = (x1 + x2) / 2.0

    contact_x = p_cur[0]
    contact_y = p_cur[1]

    lateral_offset = contact_x - px_center  

    threshold = 0.15 * player_w

    ball_on_screen_right = lateral_offset >  threshold   # larger x
    ball_on_screen_left  = lateral_offset < -threshold   # smaller x
    ball_in_middle       = abs(lateral_offset) <= threshold

    # Smash / Serve detection
    upper_body      = contact_y < (y1 + 0.45 * player_h)
    ball_above_bbox = contact_y < y1

    out_speed = speed(v_out)
    out_vy    = v_out[1]

    toward_opponent = (
        (p_side == "bottom" and out_vy < -0.5) or   
        (p_side == "top"    and out_vy >  0.5) or 
        p_side == "unknown"
    )

    is_smash_speed = out_speed >= PADEL_SMASH_SPEED_PX

    if ball_above_bbox and toward_opponent:
        return "Serve/Smash"
    if (upper_body or ball_in_middle) and toward_opponent and is_smash_speed:
        return "Serve/Smash"

    #Bottom players
    if p_side == "bottom":
        if ball_on_screen_right:
            return "Forehand"  
        elif ball_on_screen_left:
            return "Backhand"  
        else:
            return "Forehand" if v_out[0] < 0 else "Backhand"

    # ── TOP players
    elif p_side == "top":
        if ball_on_screen_left:
            return "Forehand"   
        elif ball_on_screen_right:
            return "Backhand"   
        else:
            return "Forehand" if v_out[0] > 0 else "Backhand"
        
    return "Serve/Smash"


#Classifier
def classify_shots(
    ball_detections,
    player_detections,
    fps,
    racket_detections=None,
    evidence_threshold=0.35,   
    smooth_window=5,
    min_speed=PADEL_MIN_SPEED_PX,
):

    court_midline = compute_court_midline(player_detections)

    raw_pos    = [ball_center(b) for b in ball_detections]
    raw_pos    = interpolate_missing(raw_pos, max_gap=8)

    ball_pos   = smooth_positions(raw_pos, window=smooth_window)

    valid_idx  = [i for i, p in enumerate(ball_pos) if p is not None]
    valid_set  = set(valid_idx)

    if racket_detections is None:
        racket_detections = [{} for _ in ball_detections]

    cooldown = max(2, int(PADEL_COOLDOWN_SEC * fps))   
    shots    = []
    last_hit = -999

    for i in range(2, len(ball_pos) - 2):

        if i - last_hit < cooldown:
            continue

        p_cur = ball_pos[i]
        if p_cur is None or i not in valid_set:
            continue

        prev_cands = [k for k in valid_idx if k < i]
        next_cands = [k for k in valid_idx if k > i]
        if not prev_cands or not next_cands:
            continue

        j      = prev_cands[-1]
        k      = next_cands[0]
        p_prev = ball_pos[j]
        p_next = ball_pos[k]
        if p_prev is None or p_next is None:
            continue

        dt_in  = max(1, i - j)
        dt_out = max(1, k - i)
        v_in   = ((p_cur[0]  - p_prev[0]) / dt_in,  (p_cur[1]  - p_prev[1]) / dt_in)
        v_out  = ((p_next[0] - p_cur[0])  / dt_out, (p_next[1] - p_cur[1])  / dt_out)

        # Speed floor
        if max(speed(v_in), speed(v_out)) < min_speed:
            continue

        has_turn = (
            x_direction_changed(v_in, v_out, min_turn_px=0.8)
            or direction_changed(v_in, v_out)
            or y_direction_changed(v_in, v_out, min_turn_px=0.8)
        )
        if not has_turn:
            continue

        players = player_detections[i]
        if not players:
            continue

        best_player = None
        best_score  = float("inf")
        best_dist   = float("inf")

        for pid, bbox in players.items():
            x1, y1, x2, y2 = bbox
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            expanded = (x1 - 0.6 * w, y1 - 0.5 * h, x2 + 0.6 * w, y2 + 0.5 * h)
            dist = point_to_bbox_distance(p_cur, expanded)
            if dist > 3.0 * h:
                continue
            cx    = (x1 + x2) / 2.0
            score = dist - 0.20 * abs(p_cur[0] - cx)
            if score < best_score:
                best_score  = score
                best_player = (pid, bbox)
                best_dist   = dist

        if best_player is None:
            continue

        pid, bbox = best_player

        p_side    = player_side(players, pid, court_midline)
        shot_type = classify_shot_type(p_cur, v_in, v_out, bbox, p_side)

        x1b, y1b = bbox[0], bbox[1]
        ball_above_head = p_cur[1] < y1b
        if ball_above_head and shot_type == "Serve/Smash":
            ev_total = 0.99
            ev       = {"total": ev_total, "note": "ball above head — guaranteed"}
        else:
            ev       = compute_evidence(
                v_in, v_out, p_cur, bbox, best_dist, racket_detections[i]
            )
            ev_total = ev["total"]

        if ev_total < evidence_threshold:
            continue

        shots.append({
            "frame":         i,
            "timestamp_sec": round(i / max(1, fps), 3),
            "shot_type":     shot_type,
            "player_id":     pid,
            "evidence":      round(ev_total, 4),
        })

        last_hit = i

    return shots

# Analytics
def compute_shot_analytics(shots):
    shot_counts      = Counter(s["shot_type"] for s in shots)
    player_breakdown = defaultdict(Counter)
    for s in shots:
        player_breakdown[str(s["player_id"])][s["shot_type"]] += 1

    avg_ev = (
        sum(s.get("evidence", 0) for s in shots) / len(shots) if shots else 0.0
    )

    return {
        "total_shots":      len(shots),
        "shot_counts":      dict(shot_counts),
        "avg_evidence":     round(avg_ev, 3),
        "player_breakdown": {pid: dict(tc) for pid, tc in player_breakdown.items()},
    }