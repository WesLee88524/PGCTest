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


def draw_tracking_panel(image, objects, pair, rows):
    panel = image.copy()
    tid_a, tid_b = pair
    for track_id, obj in objects.items():
        box = tlwh_to_tlbr(obj["bbox"]).astype(int)
        color = (160, 160, 160)
        thickness = 1
        if track_id in pair:
            color = color_for_id(track_id)
            thickness = 4
        cv2.rectangle(panel, tuple(box[:2]), tuple(box[2:]), color, thickness)
        cv2.putText(panel, str(track_id), (box[0], max(18, box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    if tid_a in objects and tid_b in objects:
        ca = bottom_center(objects[tid_a]["bbox"]).astype(int)
        cb = bottom_center(objects[tid_b]["bbox"]).astype(int)
        cv2.line(panel, tuple(ca), tuple(cb), (0, 230, 255), 3)

    for tid, key, color in [(tid_a, "box_a", color_for_id(tid_a)), (tid_b, "box_b", color_for_id(tid_b))]:
        pts = [bottom_center(row[key]).astype(int) for row in rows[-8:]]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            cv2.arrowedLine(panel, tuple(p0), tuple(p1), color, 2, tipLength=0.25)
    return panel


def draw_curve(canvas, rect, xs, series, labels, colors, y_range, title):
    x0, y0, w, h = rect
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (235, 235, 235), 1)
    cv2.putText(canvas, title, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2)
    for frac in [0.0, 0.5, 1.0]:
        y = int(y0 + h - frac * h)
        cv2.line(canvas, (x0, y), (x0 + w, y), (235, 235, 235), 1)
    if len(xs) <= 1:
        return
    ymin, ymax = y_range
    denom = max(ymax - ymin, 1e-6)
    x_min, x_max = min(xs), max(xs)
    x_denom = max(x_max - x_min, 1)
    for values, color in zip(series, colors):
        pts = []
        for x, value in zip(xs, values):
            px = int(x0 + (x - x_min) / x_denom * w)
            py = int(y0 + h - np.clip((value - ymin) / denom, 0.0, 1.0) * h)
            pts.append((px, py))
        for p0, p1 in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, p0, p1, color, 2)
        for pt in pts:
            cv2.circle(canvas, pt, 3, color, -1)
    lx = x0 + 12
    for label, color in zip(labels, colors):
        cv2.rectangle(canvas, (lx, y0 + 12), (lx + 18, y0 + 26), color, -1)
        cv2.putText(canvas, label, (lx + 25, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1)
        lx += 145


def make_figure(image, objects, best, video_name):
    rows = best["rows"]
    metrics = best["metrics"]
    pair = best["pair"]

    max_w = 980
    scale = min(1.0, max_w / float(image.shape[1]))
    if scale != 1.0:
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        scaled_objects = {}
        for tid, obj in objects.items():
            scaled = obj["bbox"].copy()
            scaled *= scale
            scaled_objects[tid] = {"bbox": scaled, "conf": obj["conf"]}
        scaled_rows = []
        for row in rows:
            r = dict(row)
            r["box_a"] = row["box_a"] * scale
            r["box_b"] = row["box_b"] * scale
            scaled_rows.append(r)
        objects = scaled_objects
        rows = scaled_rows

    top = draw_tracking_panel(image, objects, pair, rows)
    chart_h = 420
    pad = 32
    width = max(top.shape[1], 980)
    canvas = np.ones((top.shape[0] + chart_h + pad * 2, width, 3), dtype=np.uint8) * 255
    canvas[: top.shape[0], : top.shape[1]] = top

    title = "Crowded targets are not independent"
    subtitle = "video={} frame={} pair=({}, {}) score={:.3f} dist={:.2f} occ-sync={:.2f}".format(
        video_name,
        best["frame_id"],
        pair[0],
        pair[1],
        metrics["score"],
        metrics["mean_distance"],
        metrics["occlusion_sync"],
    )
    cv2.rectangle(canvas, (0, 0), (width, 58), (255, 255, 255), -1)
    cv2.putText(canvas, title, (24, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2)
    cv2.putText(canvas, subtitle, (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 70), 1)

    xs = [row["frame"] for row in best["rows"]]
    distances = [row["distance"] for row in best["rows"]]
    pair_ious = [row["iou"] for row in best["rows"]]
    occ_a = [row["occ_a"] for row in best["rows"]]
    occ_b = [row["occ_b"] for row in best["rows"]]
    rel = np.asarray([row["rel"] for row in best["rows"]], dtype=np.float32)
    rel_x = rel[:, 0].tolist()
    rel_y = rel[:, 1].tolist()

    y_base = top.shape[0] + pad + 10
    chart_w = (width - pad * 3) // 2
    draw_curve(canvas, (pad, y_base, chart_w, 145), xs, [distances], ["normalized distance"], [(40, 110, 220)], (0, max(3.0, max(distances) * 1.15)), "persistent spatial neighborhood")
    draw_curve(canvas, (pad * 2 + chart_w, y_base, chart_w, 145), xs, [occ_a, occ_b, pair_ious], ["target A pressure", "target B pressure", "pair IoU"], [(30, 150, 70), (210, 100, 40), (120, 80, 200)], (0, 1.0), "correlated occlusion context")
    rel_min = float(min(min(rel_x), min(rel_y)))
    rel_max = float(max(max(rel_x), max(rel_y)))
    margin = max(20.0, (rel_max - rel_min) * 0.15)
    draw_curve(canvas, (pad, y_base + 230, width - pad * 2, 145), xs, [rel_x, rel_y], ["relative x", "relative y"], [(200, 70, 70), (70, 130, 200)], (rel_min - margin, rel_max + margin), "stable relative motion can serve as collaborative reference")
    return canvas


def save_summary(path, best, video_name, image_file):
    metrics = dict(best["metrics"])
    payload = {
        "video": video_name,
        "frame_id": int(best["frame_id"]),
        "image_file": image_file,
        "track_pair": [int(best["pair"][0]), int(best["pair"][1])],
        "metrics": metrics,
        "series": [
            {
                "frame": int(row["frame"]),
                "normalized_distance": float(row["distance"]),
                "pair_iou": float(row["iou"]),
                "occlusion_pressure_a": float(row["occ_a"]),
                "occlusion_pressure_b": float(row["occ_b"]),
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
