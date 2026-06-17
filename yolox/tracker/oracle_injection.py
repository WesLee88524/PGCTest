"""
Oracle Injection Module - 在在线跟踪中注入 Ground Truth Pair 关系

通过注入 GT 来构建完美的 Pair，验证"基于 Pair 协同"的核心 idea 理论上限。

使用方法:
    1. 先用 tools/mining_oracle_pairs.py 生成 gt_oracle_pairs.pkl
    2. 在 track.py 中添加 --use_oracle_pairs --oracle_pairs_path xxx

工作原理:
    1. ID Matching: 将 Tracker Tracks 与当前帧的 GT Bounding Boxes 做 IoU 匹配
    2. Force Injection:
       - 如果 (GT_ID_i, GT_ID_j) 在 Oracle Pair Dict 中:
         强制令 A_ij = 1.0, 状态 = ACTIVE, 权重 = 1.0
       - 如果不在:
         强制令 A_ij = 0.0, 状态 = UNPAIRED
"""

import pickle
import numpy as np
from scipy.optimize import linear_sum_assignment


# Pair 状态常量（从 pgc_tracker.py 复制）
PAIR_UNPAIRED = "U"
PAIR_CANDIDATE = "C"
PAIR_ACTIVE = "A"
PAIR_WEAK = "W"


class OracleInjector:
    """
    Oracle Injection 控制器

    负责:
    1. 加载预计算的 Oracle Pair Dict
    2. 在每一帧进行 Track-GT ID 匹配
    3. 注入 Oracle 关系到 Pair 状态机
    """

    def __init__(self, oracle_pairs_path=None, enable=True):
        self.enable = enable
        self.oracle_pairs = {}  # {(gt_id_i, gt_id_j): info}  标准化为 (小, 大)
        self.track_to_gt = {}   # {track_id: gt_id}
        self.gt_to_track = {}   # {gt_id: track_id}

        if oracle_pairs_path and enable:
            self._load_oracle_pairs(oracle_pairs_path)

    def _load_oracle_pairs(self, path):
        """加载预计算的 Oracle Pair 字典"""
        try:
            with open(path, 'rb') as f:
                oracle_pairs = pickle.load(f)

            # 合并所有视频的 Oracle Pairs
            # 标准化 key 为 (min_id, max_id) 格式
            self.oracle_pairs = {}
            for video_name, pairs in oracle_pairs.items():
                for key, info in pairs.items():
                    gt_i, gt_j = info['gt_id_i'], info['gt_id_j']
                    normalized_key = tuple(sorted([gt_i, gt_j]))
                    self.oracle_pairs[normalized_key] = info

            print(f"[Oracle] Loaded {len(self.oracle_pairs)} Oracle Pairs from {path}")
        except Exception as e:
            print(f"[Oracle] Warning: Failed to load Oracle Pairs: {e}")
            self.oracle_pairs = {}

    def match_tracks_to_gt(self, tracks, gt_boxes, iou_threshold=0.5):
        """
        将 Tracker Tracks 与当前帧的 GT Bounding Boxes 做二分图匹配

        Args:
            tracks: 当前帧的 tracker tracks
            gt_boxes: dict {gt_id: (x, y, w, h)}

        Returns:
            track_to_gt: {track_id: gt_id}
            gt_to_track: {gt_id: track_id}
        """
        self.track_to_gt = {}
        self.gt_to_track = {}

        if not tracks or not gt_boxes:
            return self.track_to_gt, self.gt_to_track

        # 构建 Cost Matrix (使用 1 - IoU 作为代价)
        track_ids = [t.track_id for t in tracks]
        gt_ids = list(gt_boxes.keys())

        n_tracks = len(track_ids)
        n_gt = len(gt_ids)
        cost_matrix = np.zeros((n_tracks, n_gt))

        for i, track in enumerate(tracks):
            track_tlbr = self._tlwh_to_tlbr(track.tlwh)
            for j, gt_id in enumerate(gt_ids):
                gt_box = gt_boxes[gt_id]
                gt_tlbr = self._tlwh_to_tlbr(gt_box)
                iou = self._compute_iou(track_tlbr, gt_tlbr)
                cost_matrix[i, j] = 1.0 - iou  # 代价 = 1 - IoU

        # Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] <= (1.0 - iou_threshold):
                track_id = track_ids[i]
                gt_id = gt_ids[j]
                self.track_to_gt[track_id] = gt_id
                self.gt_to_track[gt_id] = track_id

        return self.track_to_gt, self.gt_to_track

    def get_oracle_affinity(self, track_i, track_j):
        """
        获取 Oracle 亲和度

        Returns:
            1.0 如果 (GT_ID_i, GT_ID_j) 是 Oracle Pair
            0.0 否则
        """
        if not self.enable or not self.oracle_pairs:
            return None  # 未启用 Oracle，返回 None 使用正常计算

        gt_i = self.track_to_gt.get(track_i.track_id)
        gt_j = self.track_to_gt.get(track_j.track_id)

        if gt_i is None or gt_j is None:
            return 0.0  # 至少有一个不在 GT 中

        if gt_i == gt_j:
            return 0.0  # 同一个 GT ID

        normalized_key = tuple(sorted([gt_i, gt_j]))
        if normalized_key in self.oracle_pairs:
            return 1.0
        return 0.0

    def get_oracle_state(self, track_i, track_j):
        """
        获取 Oracle 状态

        Returns:
            PAIR_ACTIVE 如果 (GT_ID_i, GT_ID_j) 是 Oracle Pair
            PAIR_UNPAIRED 否则
        """
        if not self.enable or not self.oracle_pairs:
            return None  # 未启用 Oracle

        gt_i = self.track_to_gt.get(track_i.track_id)
        gt_j = self.track_to_gt.get(track_j.track_id)

        if gt_i is None or gt_j is None:
            return PAIR_UNPAIRED

        if gt_i == gt_j:
            return PAIR_UNPAIRED

        normalized_key = tuple(sorted([gt_i, gt_j]))
        if normalized_key in self.oracle_pairs:
            return PAIR_ACTIVE
        return PAIR_UNPAIRED

    def get_oracle_attention_weight(self, track_i, track_j):
        """
        获取 Oracle Attention 权重

        Returns:
            1.0 如果是 Oracle Pair
            0.0 否则
        """
        if not self.enable or not self.oracle_pairs:
            return None

        gt_i = self.track_to_gt.get(track_i.track_id)
        gt_j = self.track_to_gt.get(track_j.track_id)

        if gt_i is None or gt_j is None:
            return 0.0

        if gt_i == gt_j:
            return 0.0

        normalized_key = tuple(sorted([gt_i, gt_j]))
        if normalized_key in self.oracle_pairs:
            return 1.0
        return 0.0

    def is_oracle_pair(self, track_i, track_j):
        """检查是否是 Oracle Pair"""
        if not self.enable or not self.oracle_pairs:
            return None

        gt_i = self.track_to_gt.get(track_i.track_id)
        gt_j = self.track_to_gt.get(track_j.track_id)

        if gt_i is None or gt_j is None:
            return False

        if gt_i == gt_j:
            return False

        normalized_key = tuple(sorted([gt_i, gt_j]))
        return normalized_key in self.oracle_pairs

    def get_oracle_pair_info(self, track_i, track_j):
        """获取 Oracle Pair 的详细信息"""
        if not self.enable or not self.oracle_pairs:
            return None

        gt_i = self.track_to_gt.get(track_i.track_id)
        gt_j = self.track_to_gt.get(track_j.track_id)

        if gt_i is None or gt_j is None:
            return None

        if gt_i == gt_j:
            return None

        normalized_key = tuple(sorted([gt_i, gt_j]))
        return self.oracle_pairs.get(normalized_key)

    @staticmethod
    def _tlwh_to_tlbr(tlwh):
        """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
        return np.array([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]])

    @staticmethod
    def _compute_iou(tlbr1, tlbr2):
        """计算两个框的 IoU"""
        x1 = max(tlbr1[0], tlbr2[0])
        y1 = max(tlbr1[1], tlbr2[1])
        x2 = min(tlbr1[2], tlbr2[2])
        y2 = min(tlbr1[3], tlbr2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        box1_area = (tlbr1[2] - tlbr1[0]) * (tlbr1[3] - tlbr1[1])
        box2_area = (tlbr2[2] - tlbr2[0]) * (tlbr2[3] - tlbr2[1])

        union_area = box1_area + box2_area - inter_area

        return inter_area / (union_area + 1e-6)

    def get_stats(self):
        """获取 Oracle 统计信息"""
        return {
            'total_oracle_pairs': len(self.oracle_pairs),
            'matched_tracks': len(self.track_to_gt),
            'enabled': self.enable,
        }


class GTBoxProvider:
    """
    GT Bounding Box 提供器

    负责加载和提供每帧的 GT Bounding Boxes
    """

    def __init__(self, gt_file_path=None):
        self.gt_data = {}  # {frame_id: {gt_id: (x, y, w, h)}}
        self.gt_file_path = gt_file_path

        if gt_file_path:
            self._load_gt()

    def _load_gt(self):
        """加载 GT 文件"""
        if not self.gt_file_path or not os.path.exists(self.gt_file_path):
            return

        self.gt_data = {}
        with open(self.gt_file_path, 'r') as f:
            for line in f:
                if line.strip() == '' or line.startswith('#'):
                    continue
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue
                try:
                    frame_id = int(parts[0])
                    gt_id = int(parts[1])
                    x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    self.gt_data.setdefault(frame_id, {})[gt_id] = (x, y, w, h)
                except (ValueError, IndexError):
                    continue

        print(f"[GTBoxProvider] Loaded GT for {len(self.gt_data)} frames")

    def get_frame_gt(self, frame_id):
        """获取指定帧的 GT Bounding Boxes"""
        return self.gt_data.get(frame_id, {})

    def has_gt(self, frame_id):
        """检查是否有指定帧的 GT"""
        return frame_id in self.gt_data


import os


# 便捷函数：创建 Oracle Injector
def create_oracle_injector(args):
    """根据命令行参数创建 Oracle Injector"""
    if not getattr(args, 'use_oracle_pairs', False):
        return None

    oracle_pairs_path = getattr(args, 'oracle_pairs_path', None)
    if not oracle_pairs_path:
        print("[Oracle] Warning: use_oracle_pairs enabled but oracle_pairs_path not specified")
        return None

    return OracleInjector(
        oracle_pairs_path=oracle_pairs_path,
        enable=True
    )


def create_gt_box_provider(args):
    """根据命令行参数创建 GT Box Provider"""
    gt_file_path = getattr(args, 'gt_file_path', None)
    if not gt_file_path:
        return None
    return GTBoxProvider(gt_file_path=gt_file_path)
