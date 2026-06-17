"""
Oracle Pair Mining - 基于 Ground Truth 离线挖掘"完美协同对"

使用方法:
    python3 tools/mining_oracle_pairs.py \
        --data_dir datasets/mot \
        --split train \
        --output gt_oracle_pairs.pkl \
        --min_overlap_frames 30 \
        --max_dist_ratio 2.5 \
        --motion_variance_thresh 0.5

输出:
    gt_oracle_pairs.pkl - 包含每个视频的 Oracle Pair 字典
    格式: {
        'video_name': {
            (gt_id_i, gt_id_j): {
                'gt_id_i': int,
                'gt_id_j': int,
                'n_overlap': int,
                'mean_dist': float,
                'std_dist': float,
                'motion_variance': float,
            }
        }
    }
"""

import argparse
import os
import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path


def parse_gt_file(gt_path):
    """
    解析 MOT 数据集的 gt.txt 文件

    格式: frame_id, track_id, x, y, w, h, conf, class_id, visibility
    """
    if not os.path.exists(gt_path):
        return None

    data = defaultdict(dict)  # {frame_id: {track_id: (x, y, w, h)}}
    with open(gt_path, 'r') as f:
        for line in f:
            if line.strip() == '' or line.startswith('#'):
                continue
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            try:
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                data[frame_id][track_id] = (x, y, w, h)
            except (ValueError, IndexError):
                continue
    return data


def compute_center_distance(box1, box2):
    """计算两个框中心点的欧氏距离"""
    cx1, cy1 = box1[0] + 0.5 * box1[2], box1[1] + 0.5 * box1[3]
    cx2, cy2 = box2[0] + 0.5 * box2[2], box2[1] + 0.5 * box2[3]
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def compute_normalized_distance(box1, box2):
    """
    计算归一化距离（按目标平均宽高归一化）
    用于尺度无关的距离度量
    """
    dist = compute_center_distance(box1, box2)
    avg_h = 0.5 * (box1[3] + box2[3])
    return dist / (avg_h + 1e-6)


def find_track_pairs_at_frame(frame_data):
    """找出同一帧中所有同时存在的目标对"""
    track_ids = list(frame_data.keys())
    pairs = []
    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):
            pairs.append((track_ids[i], track_ids[j]))
    return pairs


def mining_oracle_pairs(gt_data, min_overlap_frames=30, max_dist_ratio=2.5, motion_variance_thresh=0.5):
    """
    基于 GT 挖掘 Oracle Pairs

    提取标准:
    1. 时间共现长: N_overlap > min_overlap_frames
    2. 空间距离近: 平均归一化距离 < max_dist_ratio
    3. 相对运动稳定: 距离方差 < motion_variance_thresh
    """
    if gt_data is None:
        return {}

    frame_ids = sorted(gt_data.keys())

    # Step 1: 找出所有共现的目标对及其共现帧
    cooccurrence = defaultdict(list)  # (id_i, id_j) -> [(frame_id, dist_i_j), ...]

    for frame_id in frame_ids:
        frame_data = gt_data[frame_id]
        pairs = find_track_pairs_at_frame(frame_data)

        for gt_id_i, gt_id_j in pairs:
            box_i = frame_data[gt_id_i]
            box_j = frame_data[gt_id_j]
            norm_dist = compute_normalized_distance(box_i, box_j)
            # 使用原始距离作为运动稳定性的度量
            raw_dist = compute_center_distance(box_i, box_j)
            cooccurrence[(gt_id_i, gt_id_j)].append((frame_id, norm_dist, raw_dist))

    # Step 2: 筛选满足条件的 Oracle Pairs
    oracle_pairs = {}

    for (gt_id_i, gt_id_j), observations in cooccurrence.items():
        n_overlap = len(observations)
        if n_overlap < min_overlap_frames:
            continue

        # 计算平均归一化距离
        norm_dists = [obs[1] for obs in observations]
        mean_dist = np.mean(norm_dists)
        if mean_dist > max_dist_ratio:
            continue

        # 计算距离方差（运动稳定性指标）
        raw_dists = [obs[2] for obs in observations]
        motion_variance = np.var(raw_dists)

        if motion_variance > motion_variance_thresh:
            continue

        # 符合条件，记录为 Oracle Pair
        key = tuple(sorted([gt_id_i, gt_id_j]))
        oracle_pairs[key] = {
            'gt_id_i': gt_id_i,
            'gt_id_j': gt_id_j,
            'n_overlap': n_overlap,
            'mean_dist': float(mean_dist),
            'std_dist': float(np.std(norm_dists)),
            'motion_variance': float(motion_variance),
        }

    return oracle_pairs


def process_dataset(data_root, split='train', seqs=None, dataset_type='mot', **kwargs):
    """处理整个数据集，挖掘所有视频的 Oracle Pairs

    Args:
        data_root: 数据集根目录
        split: 数据集划分 ('train', 'test', 'val')
        seqs: 指定要处理的序列，None 则处理全部
        dataset_type: 数据集类型 ('mot', 'dancetrack')
    """
    all_oracle_pairs = {}

    # DanceTrack 目录结构
    if dataset_type.lower() == 'dancetrack':
        split_root = os.path.join(data_root, split) if os.path.exists(os.path.join(data_root, split)) else data_root
    else:
        # MOT 数据集目录结构
        split_root = os.path.join(data_root, split if 'mot' in data_root.lower() else '')
        if not os.path.exists(split_root):
            split_root = data_root

    if seqs is None:
        # 自动扫描所有视频序列
        if os.path.exists(os.path.join(split_root, 'train')):
            seq_root = os.path.join(split_root, 'train')
        elif os.path.exists(os.path.join(split_root, 'test')):
            seq_root = os.path.join(split_root, 'test')
        elif os.path.exists(os.path.join(split_root, 'val')):
            seq_root = os.path.join(split_root, 'val')
        else:
            seq_root = split_root

        seqs = [d for d in os.listdir(seq_root)
                if os.path.isdir(os.path.join(seq_root, d)) and not d.startswith('.')]

    for seq_name in seqs:
        seq_path = os.path.join(seq_root, seq_name)
        if not os.path.isdir(seq_path):
            continue

        gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
        if not os.path.exists(gt_path):
            # 尝试其他可能的路径
            for subdir in ['gt', 'GT', 'groundtruth']:
                alt_gt_path = os.path.join(seq_path, subdir, 'gt.txt')
                if os.path.exists(alt_gt_path):
                    gt_path = alt_gt_path
                    break
            else:
                print(f"Warning: No gt.txt found for {seq_name}, skipping...")
                continue

        print(f"Processing {seq_name}...")
        gt_data = parse_gt_file(gt_path)
        oracle_pairs = mining_oracle_pairs(gt_data, **kwargs)

        if oracle_pairs:
            all_oracle_pairs[seq_name] = oracle_pairs
            print(f"  Found {len(oracle_pairs)} Oracle Pairs")

    return all_oracle_pairs


def analyze_oracle_pairs(oracle_pairs):
    """分析 Oracle Pairs 的统计信息"""
    total_pairs = 0
    overlap_stats = []
    dist_stats = []
    variance_stats = []

    for seq_name, pairs in oracle_pairs.items():
        for key, info in pairs.items():
            total_pairs += 1
            overlap_stats.append(info['n_overlap'])
            dist_stats.append(info['mean_dist'])
            variance_stats.append(info['motion_variance'])

    if total_pairs > 0:
        print("\n" + "=" * 60)
        print("Oracle Pairs 统计信息")
        print("=" * 60)
        print(f"总视频数: {len(oracle_pairs)}")
        print(f"总 Pair 数: {total_pairs}")
        print(f"\n共现帧数:")
        print(f"  Min: {min(overlap_stats)}, Max: {max(overlap_stats)}, Mean: {np.mean(overlap_stats):.2f}")
        print(f"\n归一化距离:")
        print(f"  Min: {min(dist_stats):.3f}, Max: {max(dist_stats):.3f}, Mean: {np.mean(dist_stats):.3f}")
        print(f"\n运动方差:")
        print(f"  Min: {min(variance_stats):.3f}, Max: {max(variance_stats):.3f}, Mean: {np.mean(variance_stats):.3f}")
        print("=" * 60)


def make_parser():
    parser = argparse.ArgumentParser("Oracle Pair Mining")
    parser.add_argument("--data_dir", type=str, default="datasets/mot",
                       help="数据集根目录 (e.g., datasets/mot 或 datasets/DanceTrack)")
    parser.add_argument("--dataset_type", type=str, default="mot",
                       choices=["mot", "dancetrack"],
                       help="数据集类型 (默认: mot)")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "test", "val"],
                       help="数据集划分")
    parser.add_argument("--seqs", type=str, nargs='+', default=None,
                       help="指定要处理的数据序列，不指定则处理全部")
    parser.add_argument("--output", type=str, default="gt_oracle_pairs.pkl",
                       help="输出文件路径")
    parser.add_argument("--min_overlap_frames", type=int, default=30,
                       help="最小共现帧数阈值 (默认: 30)")
    parser.add_argument("--max_dist_ratio", type=float, default=2.5,
                       help="最大归一化距离阈值 (默认: 2.5)")
    parser.add_argument("--motion_variance_thresh", type=float, default=0.5,
                       help="运动方差阈值 (默认: 0.5)")
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    print("=" * 60)
    print("Oracle Pair Mining")
    print("=" * 60)
    print(f"数据集类型: {args.dataset_type}")
    print(f"数据目录: {args.data_dir}")
    print(f"数据集划分: {args.split}")
    print(f"最小共现帧数: {args.min_overlap_frames}")
    print(f"最大归一化距离: {args.max_dist_ratio}")
    print(f"运动方差阈值: {args.motion_variance_thresh}")
    print("=" * 60)

    oracle_pairs = process_dataset(
        data_root=args.data_dir,
        split=args.split,
        seqs=args.seqs,
        dataset_type=args.dataset_type,
        min_overlap_frames=args.min_overlap_frames,
        max_dist_ratio=args.max_dist_ratio,
        motion_variance_thresh=args.motion_variance_thresh
    )

    if oracle_pairs:
        # 分析统计信息
        analyze_oracle_pairs(oracle_pairs)

        # 保存结果
        output_path = args.output
        with open(output_path, 'wb') as f:
            pickle.dump(oracle_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\n结果已保存到: {output_path}")
    else:
        print("\n未找到任何 Oracle Pairs，请检查数据路径和参数设置")
