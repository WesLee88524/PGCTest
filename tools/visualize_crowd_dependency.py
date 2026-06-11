import argparse
import json
import os
from collections import defaultdict

np = None
cv2 = None


def tlwh_to_tlbr(box):
    box = np.asarray(box, dtype=np.float32).copy()
    box[2:] += box[:2]
    return box


def center(box):
    return np.asarray([box[0] + 0.5 * box[2], box[1] + 0.5 * box[3]], dtype=np.float32)


def bottom_center(box):
    return np.asarray([box[0] + 0.5 * box[2], box[1] + box[3]], dtype=np.float32)


def bbox_iou(box_a, box_b):
    a = tlwh_to_tlbr(box_a)
    b = tlwh_to_tlbr(box_b)
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def norm_distance(box_a, box_b):
    return float(
        np.linalg.norm(bottom_center(box_a) - bottom_center(box_b))
        / (0.5 * (box_a[3] + box_b[3]) + 1e-6)
    )


def read_coco_mot(ann_file):
    with open(ann_file, "r") as f:
        data = json.load(f)

    images = {int(img["id"]): img for img in data["images"]}
    videos = {int(video["id"]): video["file_name"] for video in data.get("videos", [])}
    frames = defaultdict(dict)

    anns_by_image = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("category_id", 1) != 1 or ann.get("track_id", -1) < 0:
            continue
        anns_by_image[int(ann["image_id"])].append(ann)

    for image_id, image in images.items():
        video_id = int(image["video_id"])
        frame_id = int(image["frame_id"])
        objects = {}
        for ann in anns_by_image.get(image_id, []):
            track_id = int(ann["track_id"])
            objects[track_id] = {
                "bbox": np.asarray(ann["bbox"], dtype=np.float32),
                "conf": float(ann.get("conf", 1.0)),
            }
        frames[(video_id, frame_id)] = {
            "image": image,
            "objects": objects,
        }

    video_frames = defaultdict(list)
    for video_id, frame_id in frames:
        video_frames[video_id].append(frame_id)
    for video_id in video_frames:
        video_frames[video_id] = sorted(video_frames[video_id])

    if not videos:
        for image in images.values():
            file_name = image["file_name"]
            videos[int(image["video_id"])] = file_name.split("/")[0]

    return frames, video_frames, videos


def occlusion_pressure(objects, track_id):
    obj = objects.get(track_id)
    if obj is None:
        return 0.0
    return max(
        [bbox_iou(obj["bbox"], other["bbox"]) for other_id, other in objects.items() if other_id != track_id]
        or [0.0]
    )


def pair_series(frames, video_id, frame_id, pair, window):
    tid_a, tid_b = pair
    start = max(1, frame_id - window + 1)
    rows = []
    for fid in range(start, frame_id + 1):
        frame = frames.get((video_id, fid))
        if frame is None:
            continue
        objects = frame["objects"]
        if tid_a not in objects or tid_b not in objects:
            continue
        box_a = objects[tid_a]["bbox"]
        box_b = objects[tid_b]["bbox"]
        rows.append(
            {
                "frame": fid,
                "distance": norm_distance(box_a, box_b),
                "iou": bbox_iou(box_a, box_b),
                "occ_a": occlusion_pressure(objects, tid_a),
                "occ_b": occlusion_pressure(objects, tid_b),
                "rel": center(box_b) - center(box_a),
                "box_a": box_a,
                "box_b": box_b,
            }
        )
    return rows


def score_pair(rows, window):
    if len(rows) < max(4, window // 3):
        return None
    distances = np.asarray([r["distance"] for r in rows], dtype=np.float32)
    occ_a = np.asarray([r["occ_a"] for r in rows], dtype=np.float32)
    occ_b = np.asarray([r["occ_b"] for r in rows], dtype=np.float32)
    rel = np.asarray([r["rel"] for r in rows], dtype=np.float32)
    rel_norm = rel / (np.mean([r["box_a"][3] + r["box_b"][3] for r in rows]) * 0.5 + 1e-6)
    persistence = len(rows) / float(window)
    closeness = np.exp(-float(np.mean(distances)))
    stability = np.exp(-float(np.std(rel_norm, axis=0).mean()))
    occ_sync = float(np.mean(np.minimum(occ_a, occ_b)))
    score = 0.35 * persistence + 0.30 * closeness + 0.20 * stability + 0.15 * occ_sync
    return {
        "score": float(score),
        "persistence": float(persistence),
        "mean_distance": float(np.mean(distances)),
        "distance_std": float(np.std(distances)),
        "relative_stability": float(stability),
        "occlusion_sync": float(occ_sync),
    }


def find_best_pair(frames, video_frames, video_id, tau_dist, window, stride):
    best = None
    for frame_id in video_frames[video_id][:: max(1, stride)]:
        frame = frames.get((video_id, frame_id))
        if frame is None:
            continue
        objects = frame["objects"]
        ids = sorted(objects)
        for idx, tid_a in enumerate(ids):
            for tid_b in ids[idx + 1 :]:
                if norm_distance(objects[tid_a]["bbox"], objects[tid_b]["bbox"]) > tau_dist:
                    continue
                rows = pair_series(frames, video_id, frame_id, (tid_a, tid_b), window)
                metrics = score_pair(rows, window)
                if metrics is None:
                    continue
                item = {
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "pair": (tid_a, tid_b),
                    "rows": rows,
                    "metrics": metrics,
                }
                if best is None or metrics["score"] > best["metrics"]["score"]:
                    best = item
    return best


def color_for_id(track_id):
    base = int(track_id) * 37
    return ((base * 3) % 255, (base * 7) % 255, (base * 11) % 255)


PAIR_COLOR_A = (34, 126, 247)
PAIR_COLOR_B = (239, 175, 51)
LINK_COLOR = (0, 220, 255)
INK = (38, 42, 50)
MUTED = (120, 128, 140)
GRID = (224, 229, 236)
PAPER = (248, 250, 252)
ALERT = (70, 96, 245)


def blend_rect(image, pt1, pt2, color, alpha):
    overlay = image.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)


def put_label(image, text, org, color, scale=0.55, thickness=1, pad=5):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    y = max(th + pad * 2, y)
    cv2.rectangle(image, (x, y - th - pad * 2), (x + tw + pad * 2, y + baseline), (255, 255, 255), -1)
    cv2.rectangle(image, (x, y - th - pad * 2), (x + tw + pad * 2, y + baseline), color, 2)
    cv2.putText(image, text, (x + pad, y - pad), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def dashed_line(image, p0, p1, color, thickness=3, dash=18, gap=10):
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    length = float(np.linalg.norm(p1 - p0))
    if length < 1e-6:
        return
    direction = (p1 - p0) / length
    dist = 0.0
    while dist < length:
        start = p0 + direction * dist
        end = p0 + direction * min(length, dist + dash)
        cv2.line(image, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness, cv2.LINE_AA)
        dist += dash + gap


def arrow_head(image, p0, p1, color, size=16):
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    direction = p1 - p0
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return
    direction /= norm
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    tip = p1
    left = p1 - direction * size + normal * size * 0.45
    right = p1 - direction * size - normal * size * 0.45
    cv2.fillConvexPoly(image, np.asarray([tip, left, right], dtype=np.int32), color, cv2.LINE_AA)


def fit_image(image, max_w, max_h):
    scale = min(max_w / float(image.shape[1]), max_h / float(image.shape[0]))
    size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def draw_tracking_panel(image, objects, pair, rows):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    desat = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    panel = cv2.addWeighted(image, 0.48, desat, 0.52, 0)
    blend_rect(panel, (0, 0), (panel.shape[1], panel.shape[0]), (18, 24, 32), 0.18)
    tid_a, tid_b = pair

    for track_id, obj in objects.items():
        box = tlwh_to_tlbr(obj["bbox"]).astype(int)
        if track_id in pair:
            continue
        cv2.rectangle(panel, tuple(box[:2]), tuple(box[2:]), (185, 190, 198), 1, cv2.LINE_AA)
        cv2.putText(
            panel,
            str(track_id),
            (box[0], max(16, box[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (185, 190, 198),
            1,
            cv2.LINE_AA,
        )

    if tid_a in objects and tid_b in objects:
        ca = bottom_center(objects[tid_a]["bbox"]).astype(int)
        cb = bottom_center(objects[tid_b]["bbox"]).astype(int)
        dashed_line(panel, ca, cb, LINK_COLOR, 8, dash=22, gap=12)
        dashed_line(panel, ca, cb, (255, 255, 255), 3, dash=22, gap=12)
        arrow_head(panel, ca, cb, LINK_COLOR, size=22)
        cv2.circle(panel, tuple(ca), 8, PAIR_COLOR_A, -1, cv2.LINE_AA)
        cv2.circle(panel, tuple(cb), 8, PAIR_COLOR_B, -1, cv2.LINE_AA)

    for tid, key, color in [(tid_a, "box_a", PAIR_COLOR_A), (tid_b, "box_b", PAIR_COLOR_B)]:
        pts = [bottom_center(row[key]).astype(int) for row in rows[-12:]]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            cv2.arrowedLine(panel, tuple(p0), tuple(p1), color, 2, tipLength=0.20)

    for tid, color, name in [(tid_a, PAIR_COLOR_A, "Target A"), (tid_b, PAIR_COLOR_B, "Target B")]:
        if tid not in objects:
            continue
        box = tlwh_to_tlbr(objects[tid]["bbox"]).astype(int)
        x1, y1, x2, y2 = box.tolist()
        for thickness, alpha_color in [(10, color), (4, (255, 255, 255)), (3, color)]:
            cv2.rectangle(panel, (x1, y1), (x2, y2), alpha_color, thickness, cv2.LINE_AA)
        put_label(panel, "{} #{}".format(name, tid), (x1, y1 - 8), color, scale=0.58, thickness=2)

    blend_rect(panel, (18, 18), (520, 84), (255, 255, 255), 0.82)
    cv2.putText(panel, "Persistent neighboring pair in a crowded frame", (34, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.72, INK, 2, cv2.LINE_AA)
    cv2.putText(panel, "Non-pair pedestrians are intentionally muted.", (34, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.48, MUTED, 1, cv2.LINE_AA)
    return panel


def contiguous_spans(mask, xs):
    spans = []
    start = None
    last = None
    for flag, x in zip(mask, xs):
        if flag and start is None:
            start = x
        if not flag and start is not None:
            spans.append((start, last))
            start = None
        last = x
    if start is not None:
        spans.append((start, last))
    return spans


def draw_curve(
    canvas,
    rect,
    xs,
    series,
    labels,
    colors,
    y_range,
    title,
    y_label,
    x_range,
    current_frame,
    crisis_spans=None,
    show_x_label=False,
):
    x0, y0, w, h = rect
    crisis_spans = crisis_spans or []
    cv2.rectangle(canvas, (x0, y0 - 42), (x0 + w, y0 + h + 38), (255, 255, 255), -1)
    cv2.putText(canvas, title, (x0, y0 - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.58, INK, 2, cv2.LINE_AA)
    if len(xs) <= 1:
        return
    ymin, ymax = y_range
    denom = max(ymax - ymin, 1e-6)
    x_min, x_max = x_range
    x_denom = max(x_max - x_min, 1)

    def px_at(x):
        return int(x0 + (x - x_min) / x_denom * w)

    def py_at(value):
        return int(y0 + h - np.clip((value - ymin) / denom, 0.0, 1.0) * h)

    for span_start, span_end in crisis_spans:
        sx = px_at(span_start - 0.5)
        ex = px_at(span_end + 0.5)
        blend_rect(canvas, (max(x0, sx), y0), (min(x0 + w, ex), y0 + h), (219, 233, 255), 0.58)

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = int(y0 + h - frac * h)
        cv2.line(canvas, (x0, y), (x0 + w, y), GRID, 1, cv2.LINE_AA)

    cv2.line(canvas, (x0, y0 + h), (x0 + w, y0 + h), (144, 153, 166), 1, cv2.LINE_AA)
    cv2.line(canvas, (x0, y0), (x0, y0 + h), (144, 153, 166), 1, cv2.LINE_AA)
    cv2.putText(canvas, y_label, (x0 + 6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, "{:.2f}".format(ymax), (x0 - 2, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, "{:.2f}".format(ymin), (x0 - 2, y0 + h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, MUTED, 1, cv2.LINE_AA)

    if current_frame is not None:
        cx = px_at(current_frame)
        dashed_line(canvas, (cx, y0 - 5), (cx, y0 + h + 5), ALERT, thickness=2, dash=8, gap=6)
        if show_x_label:
            put_label(canvas, "current frame {}".format(current_frame), (max(x0, cx - 92), y0 + h + 31), ALERT, scale=0.38, thickness=1, pad=4)

    for values, color in zip(series, colors):
        pts = []
        for x, value in zip(xs, values):
            pts.append((px_at(x), py_at(value)))
        if len(pts) >= 2:
            fill = np.asarray(pts + [(pts[-1][0], y0 + h), (pts[0][0], y0 + h)], dtype=np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [fill], color, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.09, canvas, 0.91, 0, canvas)
        for p0, p1 in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, p0, p1, color, 3, cv2.LINE_AA)
        mark_step = max(1, len(pts) // 7)
        for pt in pts[::mark_step]:
            cv2.circle(canvas, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 4, color, 2, cv2.LINE_AA)
    lx = x0 + w - 8
    for label, color in zip(labels, colors):
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        lx -= tw + 38
        cv2.line(canvas, (lx, y0 - 23), (lx + 22, y0 - 23), color, 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (lx + 29, y0 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, MUTED, 1, cv2.LINE_AA)
        lx -= 12
    if show_x_label:
        cv2.putText(canvas, "Frame index over the temporal window", (x0 + w - 255, y0 + h + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, MUTED, 1, cv2.LINE_AA)


def draw_metric_chip(canvas, x, y, label, value, color):
    cv2.rectangle(canvas, (x, y), (x + 230, y + 64), (255, 255, 255), -1)
    cv2.rectangle(canvas, (x, y), (x + 230, y + 64), (225, 230, 238), 1, cv2.LINE_AA)
    cv2.circle(canvas, (x + 24, y + 32), 8, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, label, (x + 43, y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.41, MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, value, (x + 43, y + 51), cv2.FONT_HERSHEY_SIMPLEX, 0.62, INK, 2, cv2.LINE_AA)


def make_figure(image, objects, best, video_name):
    rows = best["rows"]
    metrics = best["metrics"]
    pair = best["pair"]

    fig_w, fig_h = 1800, 1060
    margin = 48
    title_h = 116
    left_w = 830
    gutter = 42
    right_x = margin + left_w + gutter
    right_w = fig_w - right_x - margin
    scene_h = 690

    scene, scale = fit_image(image, left_w, scene_h)
    scaled_objects = {}
    for tid, obj in objects.items():
        scaled = obj["bbox"].copy() * scale
        scaled_objects[tid] = {"bbox": scaled, "conf": obj["conf"]}
    scaled_rows = []
    for row in rows:
        r = dict(row)
        r["box_a"] = row["box_a"] * scale
        r["box_b"] = row["box_b"] * scale
        scaled_rows.append(r)

    scene_panel = draw_tracking_panel(scene, scaled_objects, pair, scaled_rows)

    canvas = np.ones((fig_h, fig_w, 3), dtype=np.uint8) * 255
    cv2.rectangle(canvas, (0, 0), (fig_w, fig_h), PAPER, -1)

    title = "Crowded Targets Are Not Independent"
    subtitle = "A persistent neighboring pair provides measurable spatial, occlusion, and motion dependencies."
    cv2.putText(canvas, title, (margin, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.15, INK, 3, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (margin, 91), cv2.FONT_HERSHEY_SIMPLEX, 0.62, MUTED, 1, cv2.LINE_AA)
    meta = "video {} | frame {} | pair #{}-#{} | score {:.3f}".format(
        video_name, best["frame_id"], pair[0], pair[1], metrics["score"]
    )
    cv2.putText(canvas, meta, (fig_w - margin - 610, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.52, MUTED, 1, cv2.LINE_AA)

    scene_x = margin
    scene_y = title_h + 20
    canvas[scene_y : scene_y + scene_panel.shape[0], scene_x : scene_x + scene_panel.shape[1]] = scene_panel
    cv2.rectangle(
        canvas,
        (scene_x, scene_y),
        (scene_x + scene_panel.shape[1], scene_y + scene_panel.shape[0]),
        (215, 221, 230),
        1,
        cv2.LINE_AA,
    )

    xs = [row["frame"] for row in best["rows"]]
    distances = [row["distance"] for row in best["rows"]]
    pair_ious = [row["iou"] for row in best["rows"]]
    occ_a = [row["occ_a"] for row in best["rows"]]
    occ_b = [row["occ_b"] for row in best["rows"]]
    rel = np.asarray([row["rel"] for row in best["rows"]], dtype=np.float32)
    rel_x = rel[:, 0].tolist()
    rel_y = rel[:, 1].tolist()

    occ_sum = np.asarray(occ_a, dtype=np.float32) + np.asarray(occ_b, dtype=np.float32) + np.asarray(pair_ious, dtype=np.float32)
    crisis_threshold = max(0.18, float(np.percentile(occ_sum, 72)) if len(occ_sum) else 0.18)
    crisis_mask = occ_sum >= crisis_threshold
    crisis_spans = contiguous_spans(crisis_mask.tolist(), xs)
    current_frame = best["frame_id"]
    x_range = (min(xs), max(xs))

    chip_y = scene_y + scene_panel.shape[0] + 24
    draw_metric_chip(canvas, scene_x, chip_y, "persistence", "{:.0f}% of window".format(metrics["persistence"] * 100), PAIR_COLOR_A)
    draw_metric_chip(canvas, scene_x + 250, chip_y, "mean distance", "{:.2f} body heights".format(metrics["mean_distance"]), LINK_COLOR)
    draw_metric_chip(canvas, scene_x + 500, chip_y, "occlusion sync", "{:.2f}".format(metrics["occlusion_sync"]), ALERT)

    note_y = chip_y + 92
    cv2.putText(canvas, "The pair stays spatially close while sharing crowd pressure.", (scene_x, note_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, INK, 2, cv2.LINE_AA)
    cv2.putText(canvas, "This turns a local ambiguity into a usable collaborative reference.", (scene_x, note_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, MUTED, 1, cv2.LINE_AA)

    chart_h = 178
    chart_gap = 86
    chart_y = scene_y + 38
    draw_curve(
        canvas,
        (right_x, chart_y, right_w, chart_h),
        xs,
        [distances],
        ["D_ij^t"],
        [(47, 107, 214)],
        (0, max(3.0, max(distances) * 1.18)),
        "1. Persistent Spatial Proximity",
        "norm. distance",
        x_range,
        current_frame,
        crisis_spans=crisis_spans,
    )
    draw_curve(
        canvas,
        (right_x, chart_y + chart_h + chart_gap, right_w, chart_h),
        xs,
        [occ_a, occ_b, pair_ious],
        ["O_i^t (A)", "O_j^t (B)", "IoU_ij^t"],
        [(45, 156, 96), (236, 144, 54), (133, 92, 214)],
        (0, 1.0),
        "2. Correlated Occlusion Context",
        "pressure / IoU",
        x_range,
        current_frame,
        crisis_spans=crisis_spans,
    )
    rel_min = float(min(min(rel_x), min(rel_y)))
    rel_max = float(max(max(rel_x), max(rel_y)))
    margin = max(20.0, (rel_max - rel_min) * 0.15)
    draw_curve(
        canvas,
        (right_x, chart_y + 2 * (chart_h + chart_gap), right_w, chart_h),
        xs,
        [rel_x, rel_y],
        ["Delta x_ij^t", "Delta y_ij^t"],
        [(210, 75, 82), (44, 129, 196)],
        (rel_min - margin, rel_max + margin),
        "3. Stable Relative Motion",
        "pixels",
        x_range,
        current_frame,
        crisis_spans=crisis_spans,
        show_x_label=True,
    )

    if crisis_spans:
        span = crisis_spans[-1]
        cv2.rectangle(canvas, (right_x, fig_h - 96), (right_x + 454, fig_h - 48), (255, 255, 255), -1)
        cv2.rectangle(canvas, (right_x, fig_h - 96), (right_x + 454, fig_h - 48), (218, 225, 235), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (right_x + 16, fig_h - 82), (right_x + 48, fig_h - 62), (219, 233, 255), -1)
        cv2.putText(
            canvas,
            "highlighted interval: severe occlusion / detector ambiguity (frames {}-{})".format(span[0], span[1]),
            (right_x + 60, fig_h - 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            MUTED,
            1,
            cv2.LINE_AA,
        )

    return canvas


def save_summary(path, best, video_name, image_file):
    metrics = dict(best["metrics"])
    occ_sum = np.asarray(
        [row["occ_a"] + row["occ_b"] + row["iou"] for row in best["rows"]],
        dtype=np.float32,
    )
    crisis_threshold = max(0.18, float(np.percentile(occ_sum, 72)) if len(occ_sum) else 0.18)
    crisis_spans = contiguous_spans((occ_sum >= crisis_threshold).tolist(), [row["frame"] for row in best["rows"]])
    payload = {
        "video": video_name,
        "frame_id": int(best["frame_id"]),
        "image_file": image_file,
        "track_pair": [int(best["pair"][0]), int(best["pair"][1])],
        "metrics": metrics,
        "crisis_region": {
            "definition": "frames whose occlusion_pressure_a + occlusion_pressure_b + pair_iou is in the upper highlighted range",
            "threshold": float(crisis_threshold),
            "spans": [[int(start), int(end)] for start, end in crisis_spans],
        },
        "series": [
            {
                "frame": int(row["frame"]),
                "normalized_distance": float(row["distance"]),
                "pair_iou": float(row["iou"]),
                "occlusion_pressure_a": float(row["occ_a"]),
                "occlusion_pressure_b": float(row["occ_b"]),
                "relative_x": float(row["rel"][0]),
                "relative_y": float(row["rel"][1]),
            }
            for row in best["rows"]
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def make_parser():
    parser = argparse.ArgumentParser("visualize whether crowded targets are independent")
    parser.add_argument("--ann-file", default="datasets/mot/annotations/val_half.json")
    parser.add_argument("--img-root", default="datasets/mot/train")
    parser.add_argument("--output-dir", default="outputs/crowd_dependency")
    parser.add_argument("--video", default=None, help="video name or video id, e.g. MOT17-05-FRCNN")
    parser.add_argument("--frame-id", type=int, default=None, help="use a fixed frame instead of auto search")
    parser.add_argument("--track-ids", type=int, nargs=2, default=None, help="use a fixed target pair")
    parser.add_argument("--tau-dist", type=float, default=4.0)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5)
    return parser


def main():
    global cv2, np
    args = make_parser().parse_args()
    try:
        import numpy as np_module
    except ImportError as exc:
        raise RuntimeError("NumPy is required. Install numpy in the ByteTrack environment.") from exc
    try:
        import cv2 as cv2_module
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install opencv_python in the ByteTrack environment.") from exc
    np = np_module
    cv2 = cv2_module

    os.makedirs(args.output_dir, exist_ok=True)
    frames, video_frames, videos = read_coco_mot(args.ann_file)

    selected_video_ids = sorted(video_frames)
    if args.video is not None:
        selected_video_ids = [
            vid
            for vid in selected_video_ids
            if str(vid) == str(args.video) or videos.get(vid, "") == args.video
        ]
    if not selected_video_ids:
        raise RuntimeError("No matching video found in {}".format(args.ann_file))

    if args.frame_id is not None and args.track_ids is not None:
        video_id = selected_video_ids[0]
        rows = pair_series(frames, video_id, args.frame_id, tuple(args.track_ids), args.window)
        metrics = score_pair(rows, args.window)
        if metrics is None:
            raise RuntimeError("The fixed pair does not have enough valid history.")
        best = {
            "video_id": video_id,
            "frame_id": args.frame_id,
            "pair": tuple(args.track_ids),
            "rows": rows,
            "metrics": metrics,
        }
    else:
        candidates = [
            find_best_pair(frames, video_frames, video_id, args.tau_dist, args.window, args.stride)
            for video_id in selected_video_ids
        ]
        candidates = [item for item in candidates if item is not None]
        if not candidates:
            raise RuntimeError("No persistent neighboring pair was found. Try increasing --tau-dist.")
        best = max(candidates, key=lambda item: item["metrics"]["score"])

    frame = frames[(best["video_id"], best["frame_id"])]
    video_name = videos.get(best["video_id"], str(best["video_id"]))
    image_file = frame["image"]["file_name"]
    image_path = os.path.join(args.img_root, image_file)
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError("Cannot read image: {}".format(image_path))

    figure = make_figure(image, frame["objects"], best, video_name)
    figure_path = os.path.join(args.output_dir, "crowd_dependency_{}_f{:06d}.jpg".format(video_name, best["frame_id"]))
    summary_path = os.path.join(args.output_dir, "crowd_dependency_{}_f{:06d}.json".format(video_name, best["frame_id"]))
    cv2.imwrite(figure_path, figure)
    save_summary(summary_path, best, video_name, image_file)
    print("saved figure:", figure_path)
    print("saved summary:", summary_path)
    print(
        "selected video={} frame={} pair={} score={:.4f}".format(
            video_name, best["frame_id"], best["pair"], best["metrics"]["score"]
        )
    )


if __name__ == "__main__":
    main()
