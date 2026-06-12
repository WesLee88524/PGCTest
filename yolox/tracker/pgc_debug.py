import json
import os
import os.path as osp
from collections import defaultdict

import numpy as np


def _to_float(value):
    return float(np.asarray(value).item())


def _tlwh_list(tlwh):
    return [round(float(x), 3) for x in np.asarray(tlwh, dtype=float).tolist()]


def _center(tlwh):
    tlwh = np.asarray(tlwh, dtype=float)
    return np.asarray([tlwh[0] + 0.5 * tlwh[2], tlwh[1] + 0.5 * tlwh[3]], dtype=float)


def _track_id(track):
    return int(getattr(track, "track_id", -1))


def _track_state(track):
    state = getattr(track, "state", None)
    return int(state) if state is not None else -1


def summarize_tracks(tracks, topk=12):
    rows = []
    for track in tracks:
        tlwh = track.tlwh
        pred = getattr(track, "pgc_pred_tlwh", tlwh)
        shift = float(np.linalg.norm(_center(pred) - _center(tlwh)))
        rows.append(
            {
                "id": _track_id(track),
                "state": _track_state(track),
                "score": round(float(getattr(track, "score", 0.0)), 4),
                "missing": int(getattr(track, "time_since_update", 0)),
                "virtual": int(getattr(track, "virtual_update_count", 0)),
                "tlwh": _tlwh_list(tlwh),
                "pgc_pred_tlwh": _tlwh_list(pred),
                "pgc_shift_px": round(shift, 3),
                "pgc_reliability": round(float(getattr(track, "pgc_reliability", 0.0)), 4),
                "pgc_group_reliability": round(float(getattr(track, "pgc_group_reliability", 0.0)), 4),
                "pgc_occlusion": round(float(getattr(track, "pgc_occlusion", 0.0)), 4),
                "pgc_existence": round(float(getattr(track, "pgc_existence", 0.0)), 4),
                "pgc_assoc_consistency": round(float(getattr(track, "pgc_assoc_consistency", 0.0)), 4),
                "pgc_detection_confirmed": bool(getattr(track, "pgc_detection_confirmed", False)),
            }
        )
    rows.sort(key=lambda x: (x["pgc_shift_px"], x["pgc_group_reliability"], x["pgc_occlusion"]), reverse=True)
    return rows[:topk]


def summarize_cost_matrix(tracks, detections, cost, iou_cost=None, dist_cost=None, topk=20):
    if len(tracks) == 0 or len(detections) == 0 or cost.size == 0:
        return []
    rows = []
    for i, track in enumerate(tracks):
        order = np.argsort(cost[i])[: min(3, len(detections))]
        for j in order:
            item = {
                "track_id": _track_id(track),
                "det_index": int(j),
                "det_score": round(float(getattr(detections[j], "score", 0.0)), 4),
                "cost": round(float(cost[i, j]), 4),
                "det_tlwh": _tlwh_list(detections[j].tlwh),
            }
            if iou_cost is not None:
                item["iou_cost"] = round(float(iou_cost[i, j]), 4)
            if dist_cost is not None:
                item["center_cost"] = round(float(dist_cost[i, j]), 4)
            rows.append(item)
    rows.sort(key=lambda x: x["cost"])
    return rows[:topk]


def summarize_matches(matches, tracks, detections):
    rows = []
    for itrack, idet in matches:
        track = tracks[itrack]
        det = detections[idet]
        rows.append(
            {
                "track_id": _track_id(track),
                "det_index": int(idet),
                "det_score": round(float(getattr(det, "score", 0.0)), 4),
                "track_tlwh": _tlwh_list(track.tlwh),
                "det_tlwh": _tlwh_list(det.tlwh),
            }
        )
    return rows


def summarize_unmatched(indices, tracks):
    return [_track_id(tracks[i]) for i in indices]


class PGCDebugLogger(object):
    _initialized_dirs = set()
    _next_uid = 0

    def __init__(self, args):
        self.uid = PGCDebugLogger._next_uid
        PGCDebugLogger._next_uid += 1
        self.debug_dir = getattr(args, "pgc_debug_dir", None)
        self.interval = max(1, int(getattr(args, "pgc_debug_interval", 1)))
        self.topk = max(1, int(getattr(args, "pgc_debug_topk", 20)))
        self.vis = bool(getattr(args, "pgc_debug_vis", False))
        self.enabled = bool(self.debug_dir)
        self._frames = {}
        if self.enabled:
            os.makedirs(self.debug_dir, exist_ok=True)
            self.jsonl_path = osp.join(self.debug_dir, "pgc_debug.jsonl")
            self.summary_path = osp.join(self.debug_dir, "pgc_summary.json")
            self.visual_dir = osp.join(self.debug_dir, "vis")
            if self.vis:
                os.makedirs(self.visual_dir, exist_ok=True)
            if self.debug_dir not in PGCDebugLogger._initialized_dirs:
                with open(self.jsonl_path, "w"):
                    pass
                PGCDebugLogger._initialized_dirs.add(self.debug_dir)
        else:
            self.jsonl_path = None
            self.summary_path = None
            self.visual_dir = None

    def should_log(self, frame_id):
        return self.enabled and frame_id % self.interval == 0

    def log(self, event):
        if not self.should_log(int(event.get("frame_id", 0))):
            return
        event.setdefault("tracker_uid", int(self.uid))
        frame_id = int(event["frame_id"])
        self._frames.setdefault(frame_id, []).append(event)
        with open(self.jsonl_path, "a") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def flush_summary(self, frame_id, tracker):
        if not self.should_log(frame_id):
            return
        event = {
            "type": "frame_summary",
            "frame_id": int(frame_id),
            "num_tracked": len(tracker.tracked_stracks),
            "num_lost": len(tracker.lost_stracks),
            "num_removed": len(tracker.removed_stracks),
            "tracked": summarize_tracks(tracker.tracked_stracks, topk=self.topk),
            "lost": summarize_tracks(tracker.lost_stracks, topk=self.topk),
        }
        self.log(event)

    def render(self, image, tracker, frame_id):
        if not (self.enabled and self.vis and self.should_log(frame_id)):
            return None
        try:
            import cv2
        except ImportError:
            return None

        canvas = image.copy()
        colors = {
            "kf": (170, 170, 170),
            "pgc": (32, 156, 238),
            "track": (40, 190, 80),
            "lost": (70, 70, 230),
        }
        all_tracks = list(tracker.tracked_stracks) + list(tracker.lost_stracks)
        for track in all_tracks:
            tlwh = track.tlwh
            pred = getattr(track, "pgc_pred_tlwh", tlwh)
            x1, y1, w, h = [int(round(v)) for v in tlwh]
            px1, py1, pw, ph = [int(round(v)) for v in pred]
            state_color = colors["track"] if track in tracker.tracked_stracks else colors["lost"]
            cv2.rectangle(canvas, (x1, y1), (x1 + w, y1 + h), state_color, 2, cv2.LINE_AA)
            cv2.rectangle(canvas, (px1, py1), (px1 + pw, py1 + ph), colors["pgc"], 1, cv2.LINE_AA)
            c0 = (int(x1 + 0.5 * w), int(y1 + 0.5 * h))
            c1 = (int(px1 + 0.5 * pw), int(py1 + 0.5 * ph))
            cv2.arrowedLine(canvas, c0, c1, colors["pgc"], 2, cv2.LINE_AA, tipLength=0.25)
            label = "id{} g{:.2f} o{:.2f} e{:.2f}".format(
                _track_id(track),
                float(getattr(track, "pgc_group_reliability", 0.0)),
                float(getattr(track, "pgc_occlusion", 0.0)),
                float(getattr(track, "pgc_existence", 0.0)),
            )
            cv2.putText(canvas, label, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, state_color, 1, cv2.LINE_AA)

        cv2.putText(canvas, "PGC debug: green/lost boxes, blue PGC prediction", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (35, 35, 35), 2, cv2.LINE_AA)
        out_path = osp.join(self.visual_dir, "tracker{:03d}_{:06d}.jpg".format(self.uid, frame_id))
        cv2.imwrite(out_path, canvas)
        return out_path
