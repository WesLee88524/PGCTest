import argparse
import json
from collections import Counter, defaultdict

import numpy as np


def percentile(values, qs=(50, 75, 90, 95, 99)):
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {"p{}".format(q): round(float(np.percentile(arr, q)), 4) for q in qs}


def describe(values):
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=float)
    out = {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }
    out.update(percentile(values))
    return out


def load_events(path):
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser("summarize PGC inference debug logs")
    parser.add_argument("jsonl", help="path to pgc_debug.jsonl")
    parser.add_argument("--topk", type=int, default=15)
    args = parser.parse_args()

    event_counts = Counter()
    shift = []
    group = []
    occlusion = []
    existence = []
    raw_delta_abs = []
    pair_counts = []
    pair_states = Counter()
    virtual_reasons = Counter()
    virtual_applied = Counter()
    stage_costs = defaultdict(list)
    unmatched = defaultdict(list)
    large_shift_rows = []

    for event in load_events(args.jsonl):
        event_counts[event.get("type", "unknown")] += 1
        etype = event.get("type")
        if etype == "pgc_track_context":
            sx, sy = event.get("pgc_center_shift", [0.0, 0.0])
            mag = float(np.hypot(sx, sy))
            shift.append(mag)
            group.append(float(event.get("pgc_group_reliability", 0.0)))
            occlusion.append(float(event.get("pgc_occlusion", 0.0)))
            existence.append(float(event.get("pgc_existence", 0.0)))
            pair_counts.append(int(event.get("pair_count", 0)))
            if "raw_delta" in event:
                raw_delta_abs.extend(abs(float(x)) for x in event["raw_delta"])
            row = {
                "tracker_uid": event.get("tracker_uid"),
                "frame_id": event.get("frame_id"),
                "track_id": event.get("track_id"),
                "source": event.get("source"),
                "shift_px": round(mag, 3),
                "group": event.get("pgc_group_reliability"),
                "occ": event.get("pgc_occlusion"),
                "exist": event.get("pgc_existence"),
                "pair_count": event.get("pair_count"),
            }
            large_shift_rows.append(row)
        elif etype == "pgc_pairs":
            pair_states.update(event.get("pair_state_counts", {}))
        elif etype == "virtual_maintenance":
            virtual_reasons[event.get("reason", "unknown")] += 1
            virtual_applied[str(bool(event.get("applied")))] += 1
        elif etype == "association":
            stage = event.get("stage", "unknown")
            stage_costs[stage].extend(float(row["cost"]) for row in event.get("top_costs", []))
        elif etype == "matches":
            stage = event.get("stage", "unknown")
            unmatched[stage].append(len(event.get("unmatched_track_ids", [])))

    large_shift_rows.sort(key=lambda x: x["shift_px"], reverse=True)

    print("event_counts:", dict(event_counts))
    print("pgc_shift_px:", describe(shift))
    print("group_reliability:", describe(group))
    print("occlusion:", describe(occlusion))
    print("existence:", describe(existence))
    print("pair_count_per_context:", describe(pair_counts))
    if raw_delta_abs:
        print("abs_raw_delta:", describe(raw_delta_abs))
    print("pair_state_counts:", dict(pair_states))
    print("virtual_reasons:", dict(virtual_reasons))
    print("virtual_applied:", dict(virtual_applied))
    for stage, values in sorted(stage_costs.items()):
        print("cost_{}:".format(stage), describe(values))
    for stage, values in sorted(unmatched.items()):
        print("unmatched_tracks_{}:".format(stage), describe(values))
    print("largest_shifts:")
    for row in large_shift_rows[: args.topk]:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()

