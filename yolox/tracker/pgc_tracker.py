from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from yolox.tracker import matching
from .basetrack import TrackState


PAIR_UNPAIRED = "U"
PAIR_CANDIDATE = "C"
PAIR_ACTIVE = "A"
PAIR_WEAK = "W"


@dataclass
class PairState:
    state: str = PAIR_UNPAIRED
    affinity: float = 0.0
    assoc_consistency: float = 0.0
    frozen: bool = False
    on_count: int = 0
    off_count: int = 0
    last_frame: int = 0
    descriptors: deque = field(default_factory=deque)


class PGCRelationManager(object):
    """Online pair/group context used by PGCTrack.

    This is the inference-time version of the method: relation memories are
    explicit descriptor queues instead of learned Transformer features, so it
    can run on top of an existing ByteTrack checkpoint.
    """

    def __init__(self, args=None):
        self.lambda_dist = getattr(args, "pgc_lambda_dist", 0.35)
        self.lambda_scale = getattr(args, "pgc_lambda_scale", 0.20)
        self.lambda_motion = getattr(args, "pgc_lambda_motion", 0.20)
        self.lambda_quality = getattr(args, "pgc_lambda_quality", 0.25)
        self.alpha = getattr(args, "pgc_smooth_alpha", 0.75)
        self.tau_dist = getattr(args, "pgc_tau_dist", 4.0)
        self.tau_on = getattr(args, "pgc_tau_on", 0.48)
        self.tau_weak = getattr(args, "pgc_tau_weak", 0.30)
        self.tau_react = getattr(args, "pgc_tau_react", 0.42)
        self.tau_assoc = getattr(args, "pgc_tau_assoc", 0.35)
        self.k_on = getattr(args, "pgc_k_on", 2)
        self.k_off = getattr(args, "pgc_k_off", 8)
        self.k_max = getattr(args, "pgc_k_max", 5)
        self.memory_len = getattr(args, "pgc_memory_len", 8)
        self.max_stale = getattr(args, "pgc_max_stale", 30)
        self.residual_weight = getattr(args, "pgc_residual_weight", 0.12)
        self.occ_overlap_tau = getattr(args, "pgc_occ_overlap_tau", 0.08)
        self.frame_width = 1.0
        self.frame_height = 1.0
        self.pairs = {}
        self.model = None
        self.device = None
        self.debug_logger = None
        ckpt = getattr(args, "pgc_ckpt", None)
        if ckpt:
            self._load_model(ckpt, args)

    def _load_model(self, ckpt_path, args=None):
        try:
            import torch
            from .pgc_model import PGCTrackNet
        except ImportError:
            return
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
        hidden_dim = int(model_args.get("hidden_dim", getattr(args, "pgc_hidden_dim", 128)))
        memory_len = int(model_args.get("memory_len", self.memory_len))
        self.memory_len = memory_len
        self.device = torch.device(getattr(args, "pgc_device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = PGCTrackNet(hidden_dim=hidden_dim, max_len=memory_len)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _pair_key(track_a, track_b):
        return tuple(sorted((track_a.track_id, track_b.track_id)))

    @staticmethod
    def _center(tlwh):
        return np.asarray([tlwh[0] + 0.5 * tlwh[2], tlwh[1] + 0.5 * tlwh[3]], dtype=float)

    @staticmethod
    def _bottom_center(tlwh):
        return np.asarray([tlwh[0] + 0.5 * tlwh[2], tlwh[1] + tlwh[3]], dtype=float)

    @staticmethod
    def _velocity(track):
        if track.mean is None or len(track.mean) < 6:
            return np.zeros(2, dtype=float)
        return np.asarray(track.mean[4:6], dtype=float)

    @staticmethod
    def track_reliability(track):
        age = max(1, track.frame_id - track.start_frame + 1)
        age_score = min(1.0, age / 10.0)
        score = float(np.clip(getattr(track, "score", 0.0), 0.0, 1.0))
        missing = max(0, getattr(track, "time_since_update", 0))
        missing_score = np.exp(-missing / 6.0)
        tracked_bonus = 1.0 if track.state == TrackState.Tracked else 0.65
        return float(np.clip((0.50 * score + 0.30 * age_score + 0.20 * missing_score) * tracked_bonus, 0.0, 1.0))

    @staticmethod
    def missing_indicator(track):
        return 0.0 if track.state == TrackState.Tracked and getattr(track, "time_since_update", 0) == 0 else 1.0

    @staticmethod
    def association_consistency(track_i, track_j):
        assoc_i = float(np.clip(getattr(track_i, "pgc_assoc_consistency", 0.0), 0.0, 1.0))
        assoc_j = float(np.clip(getattr(track_j, "pgc_assoc_consistency", 0.0), 0.0, 1.0))
        confirmed_i = bool(getattr(track_i, "pgc_detection_confirmed", False))
        confirmed_j = bool(getattr(track_j, "pgc_detection_confirmed", False))
        return float(min(assoc_i, assoc_j)) if confirmed_i and confirmed_j else 0.0

    @staticmethod
    def pair_update_frozen(track_i, track_j):
        return (
            track_i.state != TrackState.Tracked
            or track_j.state != TrackState.Tracked
            or not bool(getattr(track_i, "pgc_detection_confirmed", False))
            or not bool(getattr(track_j, "pgc_detection_confirmed", False))
            or getattr(track_i, "virtual_update_count", 0) > 0
            or getattr(track_j, "virtual_update_count", 0) > 0
        )

    def _affinity_terms(self, track_i, track_j):
        tlwh_i = track_i.tlwh
        tlwh_j = track_j.tlwh
        eps = 1e-6

        bottom_i = self._bottom_center(tlwh_i)
        bottom_j = self._bottom_center(tlwh_j)
        norm_dist = np.linalg.norm(bottom_i - bottom_j) / ((tlwh_i[3] + tlwh_j[3]) * 0.5 + eps)
        dist_aff = np.exp(-norm_dist)

        scale_aff = np.exp(
            -abs(np.log((tlwh_i[3] + eps) / (tlwh_j[3] + eps)))
            -abs(np.log((tlwh_i[2] + eps) / (tlwh_j[2] + eps)))
        )

        vel_i = self._velocity(track_i)
        vel_j = self._velocity(track_j)
        motion_cos = float(np.dot(vel_i, vel_j) / (np.linalg.norm(vel_i) * np.linalg.norm(vel_j) + eps))
        motion_aff = 0.5 * (1.0 + np.clip(motion_cos, -1.0, 1.0))

        quality_aff = np.sqrt(self.track_reliability(track_i) * self.track_reliability(track_j))
        affinity = (
            self.lambda_dist * dist_aff
            + self.lambda_scale * scale_aff
            + self.lambda_motion * motion_aff
            + self.lambda_quality * quality_aff
        )
        return float(np.clip(affinity, 0.0, 1.0)), float(norm_dist), motion_cos

    def _descriptor(self, track_i, track_j, motion_cos):
        tlwh_i = track_i.tlwh
        tlwh_j = track_j.tlwh
        eps = 1e-6
        center_i = self._center(tlwh_i)
        center_j = self._center(tlwh_j)
        delta_pos = [
            (center_j[0] - center_i[0]) / ((tlwh_i[2] + tlwh_j[2]) * 0.5 + eps),
            (center_j[1] - center_i[1]) / ((tlwh_i[3] + tlwh_j[3]) * 0.5 + eps),
        ]
        delta_scale = [
            np.log((tlwh_j[2] + eps) / (tlwh_i[2] + eps)),
            np.log((tlwh_j[3] + eps) / (tlwh_i[3] + eps)),
        ]
        iou = 0.0
        ious = matching.ious([track_i.tlbr], [track_j.tlbr])
        if ious.size:
            iou = float(ious[0, 0])
        return np.asarray(
            delta_pos
            + delta_scale
            + [
                iou,
                motion_cos,
                self.track_reliability(track_i),
                self.track_reliability(track_j),
                self.missing_indicator(track_i),
                self.missing_indicator(track_j),
            ],
            dtype=float,
        )

    def update(self, tracks, frame_id, img_info=None):
        if img_info is not None:
            self.frame_height = max(float(img_info[0]), 1.0)
            self.frame_width = max(float(img_info[1]), 1.0)

        valid_tracks = [t for t in tracks if getattr(t, "track_id", 0) > 0 and t.mean is not None]
        for track in valid_tracks:
            track.pgc_reliability = self.track_reliability(track)
            track.pgc_group_reliability = 0.0
            track.pgc_occlusion = 0.0
            track.pgc_existence = track.pgc_reliability
            track.pgc_pred_tlwh = track.tlwh.copy()

        candidates = []
        for i, track_i in enumerate(valid_tracks):
            for track_j in valid_tracks[i + 1:]:
                affinity, norm_dist, motion_cos = self._affinity_terms(track_i, track_j)
                if norm_dist < self.tau_dist:
                    assoc = self.association_consistency(track_i, track_j)
                    frozen = self.pair_update_frozen(track_i, track_j)
                    candidates.append((track_i, track_j, affinity, norm_dist, motion_cos, assoc, frozen))

        selected_keys = set()
        per_track = defaultdict(list)
        for item in candidates:
            track_i, track_j, affinity, _, _, assoc, frozen = item
            if frozen or assoc < self.tau_assoc:
                continue
            per_track[track_i.track_id].append((affinity, item))
            per_track[track_j.track_id].append((affinity, item))
        for items in per_track.values():
            items.sort(key=lambda x: x[0], reverse=True)
            for _, item in items[: self.k_max]:
                selected_keys.add(self._pair_key(item[0], item[1]))

        current_candidate_keys = set()
        track_by_id = {t.track_id: t for t in valid_tracks}
        frozen_keys = set()
        for track_i, track_j, affinity, _, motion_cos, assoc, frozen in candidates:
            key = self._pair_key(track_i, track_j)
            if frozen:
                frozen_keys.add(key)
                continue
            if key not in selected_keys:
                continue
            current_candidate_keys.add(key)
            state = self.pairs.setdefault(key, PairState(descriptors=deque(maxlen=self.memory_len)))
            state.frozen = False
            state.affinity = self.alpha * state.affinity + (1.0 - self.alpha) * affinity
            state.assoc_consistency = self.alpha * state.assoc_consistency + (1.0 - self.alpha) * assoc
            state.last_frame = frame_id
            self._advance_lifecycle(state, seen=True, assoc_consistent=assoc >= self.tau_assoc)
            if state.state in (PAIR_ACTIVE, PAIR_WEAK):
                state.descriptors.append(self._descriptor(track_i, track_j, motion_cos))

        for key, state in list(self.pairs.items()):
            if key in current_candidate_keys or key in frozen_keys:
                state.frozen = True
                continue
            if key[0] in track_by_id and key[1] in track_by_id and self.pair_update_frozen(track_by_id[key[0]], track_by_id[key[1]]):
                state.frozen = True
                continue
            state.frozen = False
            state.affinity = self.alpha * state.affinity
            state.assoc_consistency = self.alpha * state.assoc_consistency
            self._advance_lifecycle(state, seen=False, assoc_consistent=False)
            if frame_id - state.last_frame > self.max_stale or state.state == PAIR_UNPAIRED:
                del self.pairs[key]

        self._debug_log_pairs(frame_id, valid_tracks, candidates, selected_keys)
        self._apply_group_context(valid_tracks, track_by_id, frame_id)

    def _advance_lifecycle(self, state, seen, assoc_consistent=True):
        active_observation = seen and assoc_consistent and state.affinity > self.tau_on
        if active_observation:
            state.on_count += 1
            state.off_count = 0
        else:
            state.on_count = 0
            if state.state in (PAIR_ACTIVE, PAIR_WEAK):
                state.off_count += 1

        if state.state == PAIR_UNPAIRED and active_observation:
            state.state = PAIR_CANDIDATE
        if state.state == PAIR_CANDIDATE:
            if state.on_count >= self.k_on:
                state.state = PAIR_ACTIVE
            elif not seen or not assoc_consistent or state.affinity <= self.tau_weak:
                state.state = PAIR_UNPAIRED
        elif state.state == PAIR_ACTIVE and state.affinity < self.tau_weak:
            state.state = PAIR_WEAK
            state.off_count = 1
        elif state.state == PAIR_WEAK:
            if seen and assoc_consistent and state.affinity > self.tau_react:
                state.state = PAIR_ACTIVE
                state.off_count = 0
            elif state.off_count >= self.k_off:
                state.state = PAIR_UNPAIRED

    def _memory_vector(self, state):
        if not state.descriptors:
            return None
        desc = np.asarray(state.descriptors, dtype=float)
        weights = np.linspace(0.5, 1.0, num=len(desc), dtype=float)
        return np.average(desc, axis=0, weights=weights)

    def _apply_group_context(self, tracks, track_by_id, frame_id):
        if self.model is not None:
            self._apply_learned_group_context(tracks, track_by_id, frame_id)
            return

        pair_items = defaultdict(list)
        for key, state in self.pairs.items():
            if state.frozen or state.state not in (PAIR_ACTIVE, PAIR_WEAK):
                continue
            memory = self._memory_vector(state)
            if memory is None:
                continue
            tid_a, tid_b = key
            if tid_a in track_by_id and tid_b in track_by_id:
                pair_items[tid_a].append((tid_b, state, memory, 1.0))
                mirror = memory.copy()
                mirror[0:2] *= -1.0
                mirror[2:4] *= -1.0
                mirror[6], mirror[7] = mirror[7], mirror[6]
                mirror[8], mirror[9] = mirror[9], mirror[8]
                pair_items[tid_b].append((tid_a, state, mirror, 1.0))

        for track in tracks:
            items = pair_items.get(track.track_id, [])
            if not items:
                continue
            memories = np.asarray([item[2] for item in items], dtype=float)
            affinities = np.asarray([item[1].affinity for item in items], dtype=float)
            logits = affinities - 0.35 * np.linalg.norm(memories[:, 0:2], axis=1)
            logits -= logits.max()
            alpha = np.exp(logits)
            alpha /= alpha.sum() + 1e-12
            gates = 1.0 / (1.0 + np.exp(-(4.0 * affinities + memories[:, 6] + memories[:, 7] - 3.0)))
            weighted = alpha * gates
            group_reliability = float(np.clip(weighted.sum(), 0.0, 1.0))
            context = np.sum(memories * weighted[:, None], axis=0)

            crowd_overlap = float(np.clip(np.max(memories[:, 4]) if len(memories) else 0.0, 0.0, 1.0))
            missing = self.missing_indicator(track)
            occlusion = 1.0 / (
                1.0
                + np.exp(
                    -(
                        4.0 * (crowd_overlap - self.occ_overlap_tau)
                        + 2.0 * group_reliability
                        + 1.5 * missing
                        - 1.4 * self.track_reliability(track)
                    )
                )
            )
            existence = self.track_reliability(track) * (0.60 + 0.40 * group_reliability)

            pred_tlwh = track.tlwh.copy()
            scale = np.asarray([pred_tlwh[2], pred_tlwh[3]], dtype=float)
            velocity = self._velocity(track)
            residual = -self.residual_weight * occlusion * group_reliability * context[0:2] * scale
            residual += 0.05 * group_reliability * velocity
            pred_tlwh[0:2] += residual

            track.pgc_group_context = context
            track.pgc_group_reliability = group_reliability
            track.pgc_occlusion = float(np.clip(occlusion, 0.0, 1.0))
            track.pgc_existence = float(np.clip(existence, 0.0, 1.0))
            track.pgc_pred_tlwh = pred_tlwh
            self._debug_log_track_context(
                track,
                source="heuristic",
                pair_count=len(items),
                context=context,
                residual=np.asarray([residual[0], residual[1], 0.0, 0.0], dtype=float),
                attention=alpha,
                gates=gates,
                frame_id=frame_id,
            )

    def _pair_sequence(self, state, mirror=False):
        seq = np.zeros((self.memory_len, 10), dtype=np.float32)
        mask = np.zeros((self.memory_len,), dtype=bool)
        if not state.descriptors:
            return seq, mask
        descriptors = [np.asarray(desc, dtype=np.float32).copy() for desc in list(state.descriptors)[-self.memory_len:]]
        if mirror:
            for desc in descriptors:
                desc[0:2] *= -1.0
                desc[2:4] *= -1.0
                desc[6], desc[7] = desc[7], desc[6]
                desc[8], desc[9] = desc[9], desc[8]
        start = self.memory_len - len(descriptors)
        seq[start:] = np.asarray(descriptors, dtype=np.float32)
        mask[start:] = True
        return seq, mask

    def _target_feature(self, track):
        tlwh = track.tlwh
        vel = self._velocity(track)
        return np.asarray(
            [
                tlwh[0] / self.frame_width,
                tlwh[1] / self.frame_height,
                tlwh[2] / self.frame_width,
                tlwh[3] / self.frame_height,
                vel[0] / self.frame_width,
                vel[1] / self.frame_height,
                float(np.clip(getattr(track, "score", 0.0), 0.0, 1.0)),
                self.track_reliability(track),
            ],
            dtype=np.float32,
        )

    def _apply_learned_group_context(self, tracks, track_by_id, frame_id):
        import torch

        pair_items = defaultdict(list)
        for key, state in self.pairs.items():
            if state.frozen or state.state not in (PAIR_ACTIVE, PAIR_WEAK) or not state.descriptors:
                continue
            tid_a, tid_b = key
            if tid_a in track_by_id and tid_b in track_by_id:
                pair_items[tid_a].append((state, False))
                pair_items[tid_b].append((state, True))

        active_tracks = []
        target_feats = []
        pair_seqs = []
        pair_token_masks = []
        pair_affinities = []
        pair_masks = []

        for track in tracks:
            items = pair_items.get(track.track_id, [])[: self.k_max]
            if not items:
                continue
            seq = np.zeros((self.k_max, self.memory_len, 10), dtype=np.float32)
            token_mask = np.zeros((self.k_max, self.memory_len), dtype=bool)
            affinity = np.zeros((self.k_max,), dtype=np.float32)
            pair_mask = np.zeros((self.k_max,), dtype=bool)
            for idx, (state, mirror) in enumerate(items):
                seq[idx], token_mask[idx] = self._pair_sequence(state, mirror=mirror)
                affinity[idx] = state.affinity
                pair_mask[idx] = token_mask[idx].any()
            if not pair_mask.any():
                continue
            active_tracks.append(track)
            target_feats.append(self._target_feature(track))
            pair_seqs.append(seq)
            pair_token_masks.append(token_mask)
            pair_affinities.append(affinity)
            pair_masks.append(pair_mask)

        if not active_tracks:
            return

        with torch.no_grad():
            outputs = self.model(
                torch.from_numpy(np.asarray(target_feats)).to(self.device),
                torch.from_numpy(np.asarray(pair_seqs)).to(self.device),
                torch.from_numpy(np.asarray(pair_token_masks)).to(self.device),
                torch.from_numpy(np.asarray(pair_affinities)).to(self.device),
                torch.from_numpy(np.asarray(pair_masks)).to(self.device),
            )
            deltas = outputs["delta"].detach().cpu().numpy()
            occlusions = torch.sigmoid(outputs["occlusion_logit"]).detach().cpu().numpy()
            existences = torch.sigmoid(outputs["existence_logit"]).detach().cpu().numpy()
            reliabilities = outputs["group_reliability"].detach().cpu().numpy()
            attentions = outputs["attention"].detach().cpu().numpy()
            gates = outputs["gates"].detach().cpu().numpy()
            pair_logits = outputs["pair_logits"].detach().cpu().numpy()

        for idx, track in enumerate(active_tracks):
            tlwh = track.tlwh.copy()
            norm = np.asarray([tlwh[2], tlwh[3], tlwh[2], tlwh[3]], dtype=np.float32)
            residual = np.clip(deltas[idx], -1.0, 1.0) * norm
            pred_tlwh = tlwh + residual
            pred_tlwh[2:] = np.maximum(pred_tlwh[2:], 1.0)
            track.pgc_pred_tlwh = pred_tlwh
            track.pgc_occlusion = float(np.clip(occlusions[idx], 0.0, 1.0))
            track.pgc_existence = float(np.clip(existences[idx], 0.0, 1.0))
            track.pgc_group_reliability = float(np.clip(reliabilities[idx], 0.0, 1.0))
            self._debug_log_track_context(
                track,
                source="learned",
                pair_count=int(np.asarray(pair_masks[idx]).sum()),
                delta=deltas[idx],
                residual=residual,
                attention=attentions[idx],
                gates=gates[idx],
                pair_logits=pair_logits[idx],
                frame_id=frame_id,
            )

    def _debug_log_pairs(self, frame_id, tracks, candidates, selected_keys):
        if self.debug_logger is None or not self.debug_logger.should_log(frame_id):
            return
        pair_state_counts = defaultdict(int)
        pair_rows = []
        for key, state in self.pairs.items():
            pair_state_counts[state.state] += 1
            if state.state in (PAIR_ACTIVE, PAIR_WEAK, PAIR_CANDIDATE):
                pair_rows.append(
                    {
                        "ids": [int(key[0]), int(key[1])],
                        "state": state.state,
                        "frozen": bool(state.frozen),
                        "affinity": round(float(state.affinity), 4),
                        "assoc_consistency": round(float(state.assoc_consistency), 4),
                        "on_count": int(state.on_count),
                        "off_count": int(state.off_count),
                        "memory_len": len(state.descriptors),
                    }
                )
        pair_rows.sort(key=lambda x: x["affinity"], reverse=True)
        candidate_rows = []
        for track_i, track_j, affinity, norm_dist, motion_cos, assoc, frozen in sorted(candidates, key=lambda x: x[2], reverse=True):
            if len(candidate_rows) >= self.debug_logger.topk:
                break
            key = self._pair_key(track_i, track_j)
            candidate_rows.append(
                {
                    "ids": [int(key[0]), int(key[1])],
                    "selected": key in selected_keys,
                    "frozen": bool(frozen),
                    "affinity": round(float(affinity), 4),
                    "assoc_consistency": round(float(assoc), 4),
                    "norm_dist": round(float(norm_dist), 4),
                    "motion_cos": round(float(motion_cos), 4),
                }
            )
        self.debug_logger.log(
            {
                "type": "pgc_pairs",
                "frame_id": int(frame_id),
                "num_tracks": len(tracks),
                "num_candidates": len(candidates),
                "num_selected_keys": len(selected_keys),
                "pair_state_counts": dict(pair_state_counts),
                "top_pairs": pair_rows[: self.debug_logger.topk],
                "top_candidates": candidate_rows,
            }
        )

    def _debug_log_track_context(self, track, source, pair_count, context=None, delta=None, residual=None, attention=None, gates=None, pair_logits=None, frame_id=None):
        frame_id = int(frame_id if frame_id is not None else getattr(track, "frame_id", 0))
        if self.debug_logger is None or not self.debug_logger.should_log(frame_id):
            return
        tlwh = track.tlwh
        pred = getattr(track, "pgc_pred_tlwh", tlwh)
        center_shift = self._center(pred) - self._center(tlwh)
        event = {
            "type": "pgc_track_context",
            "frame_id": frame_id,
            "source": source,
            "track_id": int(track.track_id),
            "pair_count": int(pair_count),
            "tlwh": [round(float(x), 4) for x in tlwh.tolist()],
            "pgc_pred_tlwh": [round(float(x), 4) for x in pred.tolist()],
            "pgc_center_shift": [round(float(x), 4) for x in center_shift.tolist()],
            "pgc_reliability": round(float(getattr(track, "pgc_reliability", 0.0)), 4),
            "pgc_group_reliability": round(float(getattr(track, "pgc_group_reliability", 0.0)), 4),
            "pgc_occlusion": round(float(getattr(track, "pgc_occlusion", 0.0)), 4),
            "pgc_existence": round(float(getattr(track, "pgc_existence", 0.0)), 4),
            "pgc_assoc_consistency": round(float(getattr(track, "pgc_assoc_consistency", 0.0)), 4),
            "pgc_detection_confirmed": bool(getattr(track, "pgc_detection_confirmed", False)),
        }
        if context is not None:
            event["context"] = [round(float(x), 4) for x in np.asarray(context).tolist()]
        if delta is not None:
            event["raw_delta"] = [round(float(x), 4) for x in np.asarray(delta).tolist()]
        if residual is not None:
            event["applied_residual_tlwh"] = [round(float(x), 4) for x in np.asarray(residual).tolist()]
        if attention is not None:
            event["attention"] = [round(float(x), 4) for x in np.asarray(attention).tolist()]
        if gates is not None:
            event["gates"] = [round(float(x), 4) for x in np.asarray(gates).tolist()]
        if pair_logits is not None:
            event["pair_logits"] = [round(float(x), 4) for x in np.asarray(pair_logits).tolist()]
        self.debug_logger.log(event)
