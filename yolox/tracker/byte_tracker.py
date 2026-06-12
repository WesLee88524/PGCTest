import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState
from .pgc_tracker import PGCRelationManager
from .pgc_debug import (
    PGCDebugLogger,
    summarize_cost_matrix,
    summarize_matches,
    summarize_tracks,
    summarize_unmatched,
)

if not hasattr(np, "float"):
    np.float = float

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.time_since_update = 0
        self.virtual_update_count = 0
        self.pgc_reliability = 0.0
        self.pgc_group_reliability = 0.0
        self.pgc_occlusion = 0.0
        self.pgc_existence = 0.0
        self.pgc_pred_tlwh = self._tlwh.copy()
        self.pgc_detection_confirmed = False
        self.pgc_assoc_consistency = 1.0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        self.time_since_update += 1

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
                stracks[i].time_since_update += 1

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.time_since_update = 0
        self.virtual_update_count = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.pgc_detection_confirmed = True
        self.pgc_assoc_consistency = 1.0

    def re_activate(self, new_track, frame_id, new_id=False):
        self._update_pgc_association_consistency(new_track.tlwh)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.time_since_update = 0
        self.virtual_update_count = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.pgc_detection_confirmed = True

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.time_since_update = 0
        self.virtual_update_count = 0

        new_tlwh = new_track.tlwh
        self._update_pgc_association_consistency(new_tlwh)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.pgc_detection_confirmed = True

    def virtual_update(self, pred_tlwh, frame_id, score):
        """Maintain a track through short occlusion using an internal prediction."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.virtual_update_count += 1
        self.time_since_update = 0
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(pred_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = min(float(self.score), float(score))
        self.pgc_detection_confirmed = False
        self.pgc_assoc_consistency = 0.0

    def mark_pgc_unmatched(self):
        self.pgc_detection_confirmed = False
        self.pgc_assoc_consistency = 0.0

    def _update_pgc_association_consistency(self, det_tlwh):
        pred_tlwh = np.asarray(getattr(self, "pgc_pred_tlwh", self.tlwh), dtype=float)
        det_tlwh = np.asarray(det_tlwh, dtype=float)
        pred_center = pred_tlwh[:2] + 0.5 * pred_tlwh[2:]
        det_center = det_tlwh[:2] + 0.5 * det_tlwh[2:]
        norm = 0.5 * (pred_tlwh[3] + det_tlwh[3]) + 1e-6
        dist_score = np.exp(-np.linalg.norm(pred_center - det_center) / norm)
        iou = matching.ious([self.tlwh_to_tlbr(pred_tlwh)], [self.tlwh_to_tlbr(det_tlwh)])
        iou_score = float(iou[0, 0]) if iou.size else 0.0
        self.pgc_assoc_consistency = float(np.clip(0.5 * dist_score + 0.5 * iou_score, 0.0, 1.0))

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.use_pgc = getattr(args, "use_pgc", True)
        self.pgc = PGCRelationManager(args) if self.use_pgc else None
        self.pgc_low_beta = getattr(args, "pgc_low_beta", 0.35)
        self.pgc_virtual_occ_thresh = getattr(args, "pgc_virtual_occ_thresh", 0.55)
        self.pgc_virtual_rel_thresh = getattr(args, "pgc_virtual_rel_thresh", 0.35)
        self.pgc_virtual_max = getattr(args, "pgc_virtual_max", min(12, self.max_time_lost))
        self.pgc_debug = PGCDebugLogger(args)
        if self.pgc is not None:
            self.pgc.debug_logger = self.pgc_debug

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        if self.use_pgc:
            self.pgc.update(strack_pool, self.frame_id, img_info)
            dists, pgc_terms = pgc_association_distance(strack_pool, detections, return_components=True)
            self._pgc_log_association("first_pre_fuse", strack_pool, detections, dists, pgc_terms)
        else:
            dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        if self.use_pgc:
            self._pgc_log_association("first_post_fuse", strack_pool, detections, dists, pgc_terms)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        if self.use_pgc:
            self._pgc_log_matches("first", matches, u_track, u_detection, strack_pool, detections)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        if self.use_pgc:
            dists, pgc_terms = pgc_association_distance(r_tracked_stracks, detections_second, return_components=True)
            self._pgc_log_association("second_pre_relax", r_tracked_stracks, detections_second, dists, pgc_terms)
            dists = apply_pgc_low_conf_relaxation(dists, r_tracked_stracks, self.pgc_low_beta)
            self._pgc_log_association("second_post_relax", r_tracked_stracks, detections_second, dists, pgc_terms)
        else:
            dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        if self.use_pgc:
            self._pgc_log_matches("second", matches, u_track, u_detection_second, r_tracked_stracks, detections_second)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if self.use_pgc:
                track.mark_pgc_unmatched()
            if self.use_pgc and self._pgc_virtual_maintenance(track):
                activated_starcks.append(track)
            elif not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        if self.use_pgc:
            self.pgc_debug.flush_summary(self.frame_id, self)

        return output_stracks

    def _pgc_virtual_maintenance(self, track):
        reason = "ok"
        if track.virtual_update_count >= self.pgc_virtual_max:
            reason = "max_virtual_updates"
            self._pgc_log_virtual(track, False, reason, 0.0)
            return False
        if track.pgc_occlusion <= self.pgc_virtual_occ_thresh:
            reason = "low_occlusion"
            self._pgc_log_virtual(track, False, reason, 0.0)
            return False
        if track.pgc_group_reliability <= self.pgc_virtual_rel_thresh:
            reason = "low_group_reliability"
            self._pgc_log_virtual(track, False, reason, 0.0)
            return False
        decay = np.exp(-(track.virtual_update_count + 1) / float(max(1, self.pgc_virtual_max)))
        virtual_score = track.pgc_existence * track.pgc_occlusion * track.pgc_group_reliability * decay
        track.virtual_update(track.pgc_pred_tlwh, self.frame_id, virtual_score)
        self._pgc_log_virtual(track, True, reason, virtual_score)
        return True

    def render_pgc_debug(self, image):
        if not self.use_pgc:
            return None
        return self.pgc_debug.render(image, self, self.frame_id)

    def _pgc_log_association(self, stage, tracks, detections, dists, terms):
        if not self.pgc_debug.should_log(self.frame_id):
            return
        event = {
            "type": "association",
            "stage": stage,
            "frame_id": int(self.frame_id),
            "num_tracks": len(tracks),
            "num_detections": len(detections),
            "tracks": summarize_tracks(tracks, topk=self.pgc_debug.topk),
            "top_costs": summarize_cost_matrix(
                tracks,
                detections,
                dists,
                iou_cost=terms.get("iou_cost") if terms else None,
                dist_cost=terms.get("dist_cost") if terms else None,
                topk=self.pgc_debug.topk,
            ),
        }
        self.pgc_debug.log(event)

    def _pgc_log_matches(self, stage, matches, u_track, u_detection, tracks, detections):
        if not self.pgc_debug.should_log(self.frame_id):
            return
        self.pgc_debug.log(
            {
                "type": "matches",
                "stage": stage,
                "frame_id": int(self.frame_id),
                "matches": summarize_matches(matches, tracks, detections),
                "unmatched_track_ids": summarize_unmatched(u_track, tracks),
                "unmatched_detection_indices": [int(i) for i in u_detection],
            }
        )

    def _pgc_log_virtual(self, track, applied, reason, score):
        if not self.pgc_debug.should_log(self.frame_id):
            return
        self.pgc_debug.log(
            {
                "type": "virtual_maintenance",
                "frame_id": int(self.frame_id),
                "track_id": int(track.track_id),
                "applied": bool(applied),
                "reason": reason,
                "virtual_score": round(float(score), 4),
                "track": summarize_tracks([track], topk=1)[0],
            }
        )


def pgc_association_distance(tracks, detections, lambda_iou=0.82, lambda_dist=0.18, return_components=False):
    if len(tracks) == 0 or len(detections) == 0:
        empty = np.zeros((len(tracks), len(detections)), dtype=float)
        if return_components:
            return empty, {"iou_cost": empty.copy(), "dist_cost": empty.copy()}
        return empty

    track_tlwhs = [getattr(track, "pgc_pred_tlwh", track.tlwh) for track in tracks]
    track_tlbrs = [STrack.tlwh_to_tlbr(tlwh) for tlwh in track_tlwhs]
    det_tlbrs = [det.tlbr for det in detections]
    iou_cost = 1.0 - matching.ious(track_tlbrs, det_tlbrs)
    dist_cost = normalized_center_distance(track_tlwhs, [det.tlwh for det in detections])
    total = lambda_iou * iou_cost + lambda_dist * dist_cost
    if return_components:
        return total, {"iou_cost": iou_cost, "dist_cost": dist_cost}
    return total


def normalized_center_distance(track_tlwhs, det_tlwhs):
    cost = np.zeros((len(track_tlwhs), len(det_tlwhs)), dtype=float)
    for i, track_tlwh in enumerate(track_tlwhs):
        track_center = np.asarray([track_tlwh[0] + 0.5 * track_tlwh[2], track_tlwh[1] + 0.5 * track_tlwh[3]])
        for j, det_tlwh in enumerate(det_tlwhs):
            det_center = np.asarray([det_tlwh[0] + 0.5 * det_tlwh[2], det_tlwh[1] + 0.5 * det_tlwh[3]])
            norm = 0.5 * (track_tlwh[3] + det_tlwh[3]) + 1e-6
            cost[i, j] = min(1.0, np.linalg.norm(track_center - det_center) / norm)
    return cost


def apply_pgc_low_conf_relaxation(cost_matrix, tracks, beta):
    if cost_matrix.size == 0:
        return cost_matrix
    relaxed = cost_matrix.copy()
    for row, track in enumerate(tracks):
        factor = 1.0 - beta * track.pgc_occlusion * track.pgc_group_reliability
        relaxed[row] *= np.clip(factor, 0.55, 1.0)
    return relaxed


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
