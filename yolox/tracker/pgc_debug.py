import json
import os
import os.path as osp

import numpy as np


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


class PGCDebugLogger(object):
    def __init__(self, args):
        self.debug_dir = getattr(args, "pgc_debug_dir", None)
        self.interval = max(1, int(getattr(args, "pgc_debug_interval", 1)))
        self.topk = max(1, int(getattr(args, "pgc_debug_topk", 20)))
        self.vis = bool(getattr(args, "pgc_debug_vis", False))
        self.enabled = bool(self.debug_dir)
        if self.enabled:
            os.makedirs(self.debug_dir, exist_ok=True)
            self.jsonl_path = osp.join(self.debug_dir, "pgc_debug.jsonl")
            if not os.path.exists(self.jsonl_path):
                with open(self.jsonl_path, "w"):
                    pass
        else:
            self.jsonl_path = None

    def should_log(self, frame_id):
        return self.enabled and frame_id % self.interval == 0

    def log(self, event):
        if not self.should_log(int(event.get("frame_id", 0))):
            return
        with open(self.jsonl_path, "a") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def flush_summary(self, frame_id, tracker):
        if not self.should_log(frame_id):
            return
        self.log(
            {
                "type": "frame_summary",
                "frame_id": int(frame_id),
                "num_tracked": len(tracker.tracked_stracks),
                "num_lost": len(tracker.lost_stracks),
                "num_removed": len(tracker.removed_stracks),
                "tracked": summarize_tracks(tracker.tracked_stracks, topk=self.topk),
                "lost": summarize_tracks(tracker.lost_stracks, topk=self.topk),
            }
        )
