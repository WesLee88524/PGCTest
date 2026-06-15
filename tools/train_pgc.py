from loguru import logger

import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from yolox.data import get_yolox_datadir
from yolox.exp import get_exp
from yolox.tracker.pgc_model import PGCTrackNet, pgc_loss
from yolox.data.datasets.mot import MOTDataset


def _center_xywh(box):
    return np.asarray([box[0] + 0.5 * box[2], box[1] + 0.5 * box[3]], dtype=float)


def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


class PGCMOTPairDataset(Dataset):
    def __init__(
        self,
        mot_dataset,
        input_size,
        memory_len=8,
        k_max=5,
        pair_radius=3.5,
        occ_iou_thresh=0.3,
    ):
        self.dataset = mot_dataset
        self.input_size = input_size
        self.memory_len = int(memory_len)
        self.k_max = int(k_max)
        self.pair_radius = float(pair_radius)
        self.occ_iou_thresh = float(occ_iou_thresh)

        self.frames_by_video = self._build_frames()
        self.samples = self._build_samples()

    def _build_frames(self):
        frames_by_video = {}
        for res, img_info, file_name in self.dataset.annotations:
            height, width, frame_id, video_id, _ = img_info
            video_id = int(video_id)
            frame_id = int(frame_id)
            frame = {
                "frame_id": frame_id,
                "video_id": video_id,
                "width": float(width),
                "height": float(height),
                "file_name": file_name,
                "boxes": {},
            }
            for row in res:
                track_id = int(row[5])
                frame["boxes"][track_id] = np.asarray(row[:4], dtype=float)
            frames_by_video.setdefault(video_id, []).append(frame)

        for video_id in frames_by_video:
            frames_by_video[video_id].sort(key=lambda x: x["frame_id"])
        return frames_by_video

    def _build_samples(self):
        samples = []
        for video_id, frames in self.frames_by_video.items():
            frame_map = {frame["frame_id"]: frame for frame in frames}
            frame_ids = sorted(frame_map.keys())
            for idx in range(1, len(frame_ids) - 1):
                cur = frame_map[frame_ids[idx]]
                nxt = frame_map[frame_ids[idx + 1]]
                prev = frame_map.get(frame_ids[idx - 1], None)
                for track_id, curr_box in cur["boxes"].items():
                    next_box = nxt["boxes"].get(track_id)
                    if next_box is None:
                        continue
                    samples.append(
                        {
                            "video_id": video_id,
                            "cur_frame_id": cur["frame_id"],
                            "prev_frame_id": prev["frame_id"] if prev is not None else None,
                            "next_frame_id": nxt["frame_id"],
                            "track_id": track_id,
                        }
                    )
        return samples

    def __len__(self):
        return len(self.samples)

    def _frame_at(self, video_id, frame_id):
        for frame in self.frames_by_video[video_id]:
            if frame["frame_id"] == frame_id:
                return frame
        return None

    def _pair_sequence(self, video_id, frame_ids, target_id, other_id):
        seq = np.zeros((self.memory_len, 10), dtype=np.float32)
        mask = np.zeros((self.memory_len,), dtype=bool)
        valid_frames = []
        for frame_id in frame_ids:
            frame = self._frame_at(video_id, frame_id)
            if frame is None:
                continue
            target_box = frame["boxes"].get(target_id)
            other_box = frame["boxes"].get(other_id)
            if target_box is None or other_box is None:
                continue
            valid_frames.append((frame_id, target_box, other_box, frame))

        valid_frames = valid_frames[-self.memory_len :]
        start = self.memory_len - len(valid_frames)
        for offset, (_, target_box, other_box, frame) in enumerate(valid_frames):
            eps = 1e-6
            c_t = _center_xywh(target_box)
            c_o = _center_xywh(other_box)
            delta_pos = np.asarray(
                [
                    (c_o[0] - c_t[0]) / ((target_box[2] + other_box[2]) * 0.5 + eps),
                    (c_o[1] - c_t[1]) / ((target_box[3] + other_box[3]) * 0.5 + eps),
                ],
                dtype=np.float32,
            )
            delta_scale = np.asarray(
                [
                    np.log((other_box[2] + eps) / (target_box[2] + eps)),
                    np.log((other_box[3] + eps) / (target_box[3] + eps)),
                ],
                dtype=np.float32,
            )
            iou = np.asarray(
                _iou_xyxy(
                    [target_box[0], target_box[1], target_box[0] + target_box[2], target_box[1] + target_box[3]],
                    [other_box[0], other_box[1], other_box[0] + other_box[2], other_box[1] + other_box[3]],
                ),
                dtype=np.float32,
            )
            seq[start + offset] = np.asarray(
                [
                    delta_pos[0],
                    delta_pos[1],
                    delta_scale[0],
                    delta_scale[1],
                    iou.item(),
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
            mask[start + offset] = True
        return seq, mask

    def __getitem__(self, index):
        sample = self.samples[index]
        video_id = sample["video_id"]
        cur_frame = self._frame_at(video_id, sample["cur_frame_id"])
        next_frame = self._frame_at(video_id, sample["next_frame_id"])
        prev_frame = self._frame_at(video_id, sample["prev_frame_id"]) if sample["prev_frame_id"] is not None else None
        track_id = sample["track_id"]

        curr_box = cur_frame["boxes"][track_id]
        next_box = next_frame["boxes"][track_id]
        prev_box = prev_frame["boxes"].get(track_id) if prev_frame is not None else None
        width = max(cur_frame["width"], 1.0)
        height = max(cur_frame["height"], 1.0)

        vel = np.zeros(2, dtype=np.float32)
        if prev_box is not None:
            vel = np.asarray(
                [
                    (curr_box[0] - prev_box[0]) / width,
                    (curr_box[1] - prev_box[1]) / height,
                ],
                dtype=np.float32,
            )

        target_feat = np.asarray(
            [
                curr_box[0] / width,
                curr_box[1] / height,
                curr_box[2] / width,
                curr_box[3] / height,
                vel[0],
                vel[1],
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )

        others = []
        for other_id, other_box in cur_frame["boxes"].items():
            if other_id == track_id:
                continue
            center_dist = np.linalg.norm(_center_xywh(other_box) - _center_xywh(curr_box)) / (0.5 * (curr_box[3] + other_box[3]) + 1e-6)
            others.append((center_dist, other_id, other_box))
        others.sort(key=lambda x: x[0])
        chosen = others[: self.k_max]

        pair_seqs = np.zeros((self.k_max, self.memory_len, 10), dtype=np.float32)
        pair_token_masks = np.zeros((self.k_max, self.memory_len), dtype=bool)
        pair_affinity = np.zeros((self.k_max,), dtype=np.float32)
        pair_mask = np.zeros((self.k_max,), dtype=bool)
        pair_label = np.zeros((self.k_max,), dtype=np.float32)

        history_ids = list(range(max(1, cur_frame["frame_id"] - self.memory_len + 1), cur_frame["frame_id"] + 1))
        for idx, (center_dist, other_id, other_box) in enumerate(chosen):
            seq, mask = self._pair_sequence(video_id, history_ids, track_id, other_id)
            pair_seqs[idx] = seq
            pair_token_masks[idx] = mask
            pair_mask[idx] = bool(mask.any())
            pair_affinity[idx] = float(np.exp(-center_dist))
            pair_label[idx] = 1.0 if center_dist < self.pair_radius else 0.0

        delta = np.asarray(
            [
                (next_box[0] - curr_box[0]) / (curr_box[2] + 1e-6),
                (next_box[1] - curr_box[1]) / (curr_box[3] + 1e-6),
                (next_box[2] - curr_box[2]) / (curr_box[2] + 1e-6),
                (next_box[3] - curr_box[3]) / (curr_box[3] + 1e-6),
            ],
            dtype=np.float32,
        )

        next_others = [box for oid, box in next_frame["boxes"].items() if oid != track_id]
        max_iou = 0.0
        for other_box in next_others:
            iou = _iou_xyxy(
                [next_box[0], next_box[1], next_box[0] + next_box[2], next_box[1] + next_box[3]],
                [other_box[0], other_box[1], other_box[0] + other_box[2], other_box[1] + other_box[3]],
            )
            max_iou = max(max_iou, iou)
        occlusion = np.asarray(1.0 if max_iou >= self.occ_iou_thresh else 0.0, dtype=np.float32)
        existence = np.asarray(1.0, dtype=np.float32)

        labels = {
            "delta": delta,
            "occlusion": occlusion,
            "existence": existence,
            "pair_label": pair_label,
            "pair_mask": pair_mask.astype(np.float32),
        }
        return target_feat, pair_seqs, pair_token_masks, pair_affinity, pair_mask, labels


def collate_pgc(batch):
    target_feat, pair_seqs, pair_token_masks, pair_affinity, pair_mask, labels = zip(*batch)
    batch_dict = {
        "target_feat": torch.from_numpy(np.asarray(target_feat)),
        "pair_seq": torch.from_numpy(np.asarray(pair_seqs)),
        "pair_token_mask": torch.from_numpy(np.asarray(pair_token_masks)),
        "pair_affinity": torch.from_numpy(np.asarray(pair_affinity)),
        "pair_mask": torch.from_numpy(np.asarray(pair_mask)),
        "labels": {
            "delta": torch.from_numpy(np.asarray([x["delta"] for x in labels])),
            "occlusion": torch.from_numpy(np.asarray([x["occlusion"] for x in labels])),
            "existence": torch.from_numpy(np.asarray([x["existence"] for x in labels])),
            "pair_label": torch.from_numpy(np.asarray([x["pair_label"] for x in labels])),
            "pair_mask": torch.from_numpy(np.asarray([x["pair_mask"] for x in labels])),
        },
    }
    return batch_dict


def make_parser():
    parser = argparse.ArgumentParser("PGC training parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None)
    parser.add_argument("-f", "--exp_file", default=None, type=str)
    parser.add_argument("--data_dir", default=None, type=str, help="dataset directory relative to yolox datadir")
    parser.add_argument("--train_json", default=None, type=str, help="train annotation json")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("-c", "--ckpt", default=None, type=str)
    parser.add_argument("--save_dir", default=None, type=str)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-d", "--devices", default=None, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--num_machines", default=1, type=int)
    parser.add_argument("--machine_rank", default=0, type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0, help="limit samples for quick debugging")
    parser.add_argument("--pgc_hidden_dim", type=int, default=128)
    parser.add_argument("--pgc_num_layers", type=int, default=2)
    parser.add_argument("--pgc_num_heads", type=int, default=4)
    parser.add_argument("--pgc_memory_len", type=int, default=8)
    parser.add_argument("--pgc_k_max", type=int, default=5)
    parser.add_argument("--pgc_pair_radius", type=float, default=3.5)
    parser.add_argument("--pgc_occ_iou_thresh", type=float, default=0.3)
    parser.add_argument("--pgc_motion_weight", type=float, default=1.0)
    parser.add_argument("--pgc_occ_weight", type=float, default=1.0)
    parser.add_argument("--pgc_existence_weight", type=float, default=0.2)
    parser.add_argument("--pgc_pair_weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(exp, args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    data_root = args.data_dir or "mix_mot_ch"
    data_root = data_root if os.path.isabs(data_root) else os.path.join(get_yolox_datadir(), data_root)
    train_json = args.train_json or getattr(exp, "train_ann", "train.json")
    input_size = getattr(exp, "input_size", (800, 1440))
    max_epoch = args.epochs or getattr(exp, "max_epoch", 80)

    dataset = MOTDataset(
        data_dir=data_root,
        json_file=train_json,
        name="train" if "mix_mot" in data_root or "mot" in data_root else "",
        img_size=input_size,
        preproc=None,
    )
    pgc_dataset = PGCMOTPairDataset(
        dataset,
        input_size=input_size,
        memory_len=args.pgc_memory_len,
        k_max=args.pgc_k_max,
        pair_radius=args.pgc_pair_radius,
        occ_iou_thresh=args.pgc_occ_iou_thresh,
    )
    if args.max_samples > 0:
        pgc_dataset.samples = pgc_dataset.samples[: args.max_samples]

    loader = DataLoader(
        pgc_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_pgc,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PGCTrackNet(
        hidden_dim=args.pgc_hidden_dim,
        num_heads=args.pgc_num_heads,
        num_layers=args.pgc_num_layers,
        max_len=args.pgc_memory_len,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    start_epoch = 0
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        if args.resume and isinstance(ckpt, dict) and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt.get("epoch", 0))

    out_dir = args.save_dir or os.path.join(exp.output_dir, args.experiment_name or exp.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Training PGC model on {} samples".format(len(pgc_dataset)))

    global_step = 0
    for epoch in range(start_epoch, max_epoch):
        model.train()
        meters = {"total": 0.0, "motion": 0.0, "occlusion": 0.0, "existence": 0.0, "pair": 0.0}
        t0 = time.time()
        for it, batch in enumerate(loader, start=1):
            target_feat = batch["target_feat"].to(device)
            pair_seq = batch["pair_seq"].to(device)
            pair_token_mask = batch["pair_token_mask"].to(device)
            pair_affinity = batch["pair_affinity"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            labels = {k: v.to(device) for k, v in batch["labels"].items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(target_feat, pair_seq, pair_token_mask, pair_affinity, pair_mask)
                loss_dict = pgc_loss(
                    outputs,
                    labels,
                    weights={
                        "motion": args.pgc_motion_weight,
                        "occ": args.pgc_occ_weight,
                        "pair": args.pgc_pair_weight,
                        "existence": args.pgc_existence_weight,
                    },
                )
                loss = loss_dict["total"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            meters["total"] += float(loss.detach().cpu())
            meters["motion"] += float(loss_dict["motion"].cpu())
            meters["occlusion"] += float(loss_dict["occlusion"].cpu())
            meters["existence"] += float(loss_dict["existence"].cpu())
            meters["pair"] += float(loss_dict["pair"].cpu())
            global_step += 1

            if it % 20 == 0 or it == len(loader):
                denom = float(it)
                logger.info(
                    "epoch {} iter {}/{} loss {:.4f} motion {:.4f} occ {:.4f} exist {:.4f} pair {:.4f}".format(
                        epoch + 1,
                        it,
                        len(loader),
                        meters["total"] / denom,
                        meters["motion"] / denom,
                        meters["occlusion"] / denom,
                        meters["existence"] / denom,
                        meters["pair"] / denom,
                    )
                )

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": vars(args),
        }
        if (epoch + 1) % args.save_interval == 0:
            torch.save(ckpt, os.path.join(out_dir, "latest_pgc_ckpt.pth.tar"))
        torch.save(ckpt, os.path.join(out_dir, "last_pgc_ckpt.pth.tar"))
        logger.info("epoch {} done in {:.1f}s".format(epoch + 1, time.time() - t0))

    logger.info("training finished, checkpoints saved to {}".format(out_dir))


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(getattr(args, "opts", []))
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    main(exp, args)
