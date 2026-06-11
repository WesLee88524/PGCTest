import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from yolox.tracker.pgc_model import PGCTrackNet, pgc_loss


def tlwh_to_tlbr(box):
    box = np.asarray(box, dtype=np.float32).copy()
    box[2:] += box[:2]
    return box


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


def center(box):
    return np.asarray([box[0] + 0.5 * box[2], box[1] + 0.5 * box[3]], dtype=np.float32)


def bottom_center(box):
    return np.asarray([box[0] + 0.5 * box[2], box[1] + box[3]], dtype=np.float32)


class PGCMOTDataset(Dataset):
    def __init__(
        self,
        ann_file,
        memory_len=8,
        max_neighbors=5,
        future_len=5,
        tau_neighbor=4.0,
        tau_pair=4.0,
        tau_persist=0.6,
        tau_occ_iou=0.35,
        max_samples=None,
    ):
        self.memory_len = memory_len
        self.max_neighbors = max_neighbors
        self.future_len = future_len
        self.tau_neighbor = tau_neighbor
        self.tau_pair = tau_pair
        self.tau_persist = tau_persist
        self.tau_occ_iou = tau_occ_iou

        with open(ann_file, "r") as f:
            data = json.load(f)

        self.images = {img["id"]: img for img in data["images"]}
        anns_by_image = defaultdict(list)
        for ann in data["annotations"]:
            if ann.get("category_id", 1) != 1 or ann.get("track_id", -1) < 0:
                continue
            anns_by_image[ann["image_id"]].append(ann)

        self.frames = defaultdict(dict)
        self.video_frames = defaultdict(list)
        for image_id, image in self.images.items():
            video_id = image["video_id"]
            frame_id = image["frame_id"]
            objects = {}
            for ann in anns_by_image.get(image_id, []):
                track_id = int(ann["track_id"])
                objects[track_id] = {
                    "bbox": np.asarray(ann["bbox"], dtype=np.float32),
                    "conf": float(ann.get("conf", 1.0)),
                    "vis": ann.get("visibility", ann.get("vis", None)),
                }
            self.frames[(video_id, frame_id)] = {
                "width": float(image["width"]),
                "height": float(image["height"]),
                "objects": objects,
            }
            self.video_frames[video_id].append(frame_id)

        for video_id in self.video_frames:
            self.video_frames[video_id] = sorted(set(self.video_frames[video_id]))

        self.samples = []
        for (video_id, frame_id), frame in self.frames.items():
            next_frame = self.frames.get((video_id, frame_id + 1))
            if next_frame is None:
                continue
            for track_id in frame["objects"]:
                if track_id in next_frame["objects"]:
                    self.samples.append((video_id, frame_id, track_id))

        self.samples.sort()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def _object(self, video_id, frame_id, track_id):
        frame = self.frames.get((video_id, frame_id))
        if frame is None:
            return None
        return frame["objects"].get(track_id)

    def _velocity(self, video_id, frame_id, track_id):
        obj = self._object(video_id, frame_id, track_id)
        prev = self._object(video_id, frame_id - 1, track_id)
        if obj is None or prev is None:
            return np.zeros(2, dtype=np.float32)
        return center(obj["bbox"]) - center(prev["bbox"])

    def _track_age(self, video_id, frame_id, track_id):
        age = 0
        for past_frame in range(frame_id, 0, -1):
            if self._object(video_id, past_frame, track_id) is None:
                break
            age += 1
        return age

    def _track_reliability(self, video_id, frame_id, track_id):
        obj = self._object(video_id, frame_id, track_id)
        if obj is None:
            return 0.0
        age_score = min(1.0, self._track_age(video_id, frame_id, track_id) / 10.0)
        score = float(np.clip(obj["conf"], 0.0, 1.0))
        missing_score = 1.0
        return float(np.clip(0.50 * score + 0.30 * age_score + 0.20 * missing_score, 0.0, 1.0))

    def _target_feat(self, video_id, frame_id, track_id):
        frame = self.frames[(video_id, frame_id)]
        obj = frame["objects"][track_id]
        box = obj["bbox"]
        vel = self._velocity(video_id, frame_id, track_id)
        return np.asarray(
            [
                box[0] / frame["width"],
                box[1] / frame["height"],
                box[2] / frame["width"],
                box[3] / frame["height"],
                vel[0] / frame["width"],
                vel[1] / frame["height"],
                obj["conf"],
                self._track_reliability(video_id, frame_id, track_id),
            ],
            dtype=np.float32,
        )

    def _descriptor_at(self, video_id, frame_id, track_i, track_j):
        obj_i = self._object(video_id, frame_id, track_i)
        obj_j = self._object(video_id, frame_id, track_j)
        if obj_i is None or obj_j is None:
            return None
        box_i = obj_i["bbox"]
        box_j = obj_j["bbox"]
        eps = 1e-6
        ci = center(box_i)
        cj = center(box_j)
        vi = self._velocity(video_id, frame_id, track_i)
        vj = self._velocity(video_id, frame_id, track_j)
        motion = float(np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + eps))
        return np.asarray(
            [
                (cj[0] - ci[0]) / ((box_i[2] + box_j[2]) * 0.5 + eps),
                (cj[1] - ci[1]) / ((box_i[3] + box_j[3]) * 0.5 + eps),
                math.log((box_j[2] + eps) / (box_i[2] + eps)),
                math.log((box_j[3] + eps) / (box_i[3] + eps)),
                bbox_iou(box_i, box_j),
                np.clip(motion, -1.0, 1.0),
                self._track_reliability(video_id, frame_id, track_i),
                self._track_reliability(video_id, frame_id, track_j),
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

    def _norm_distance(self, box_i, box_j):
        return np.linalg.norm(bottom_center(box_i) - bottom_center(box_j)) / ((box_i[3] + box_j[3]) * 0.5 + 1e-6)

    def _neighbors(self, video_id, frame_id, track_id):
        frame = self.frames[(video_id, frame_id)]
        box_i = frame["objects"][track_id]["bbox"]
        neighbors = []
        for other_id, obj in frame["objects"].items():
            if other_id == track_id:
                continue
            dist = self._norm_distance(box_i, obj["bbox"])
            if dist < self.tau_neighbor:
                neighbors.append((dist, other_id))
        neighbors.sort(key=lambda x: x[0])
        return [track_id for _, track_id in neighbors[: self.max_neighbors]]

    def _pair_label(self, video_id, frame_id, track_i, track_j):
        stable = 0
        total = 0
        for offset in range(1, self.future_len + 1):
            obj_i = self._object(video_id, frame_id + offset, track_i)
            obj_j = self._object(video_id, frame_id + offset, track_j)
            if obj_i is None or obj_j is None:
                continue
            total += 1
            if self._norm_distance(obj_i["bbox"], obj_j["bbox"]) < self.tau_pair:
                stable += 1
        if total == 0:
            return 0.0
        return float((stable / total) >= self.tau_persist)

    def _occlusion_label(self, video_id, frame_id, track_id):
        obj = self._object(video_id, frame_id, track_id)
        if obj is None:
            return 1.0
        if obj["vis"] is not None:
            return float(float(obj["vis"]) < 0.5)
        frame = self.frames[(video_id, frame_id)]
        max_iou = 0.0
        for other_id, other in frame["objects"].items():
            if other_id == track_id:
                continue
            max_iou = max(max_iou, bbox_iou(obj["bbox"], other["bbox"]))
        return float(max_iou > self.tau_occ_iou)

    def __getitem__(self, index):
        video_id, frame_id, track_id = self.samples[index]
        frame = self.frames[(video_id, frame_id)]
        obj = frame["objects"][track_id]
        next_obj = self.frames[(video_id, frame_id + 1)]["objects"][track_id]
        box = obj["bbox"]
        next_box = next_obj["bbox"]
        norm = np.asarray([box[2], box[3], box[2], box[3]], dtype=np.float32) + 1e-6
        delta = (next_box - box) / norm

        pair_seq = np.zeros((self.max_neighbors, self.memory_len, 10), dtype=np.float32)
        pair_token_mask = np.zeros((self.max_neighbors, self.memory_len), dtype=np.bool_)
        pair_affinity = np.zeros((self.max_neighbors,), dtype=np.float32)
        pair_label = np.zeros((self.max_neighbors,), dtype=np.float32)
        pair_mask = np.zeros((self.max_neighbors,), dtype=np.bool_)

        for neighbor_index, neighbor_id in enumerate(self._neighbors(video_id, frame_id, track_id)):
            descriptors = []
            for past_frame in range(frame_id - self.memory_len + 1, frame_id + 1):
                desc = self._descriptor_at(video_id, past_frame, track_id, neighbor_id)
                if desc is not None:
                    descriptors.append(desc)
            if not descriptors:
                continue
            descriptors = descriptors[-self.memory_len :]
            start = self.memory_len - len(descriptors)
            pair_seq[neighbor_index, start:] = np.asarray(descriptors, dtype=np.float32)
            pair_token_mask[neighbor_index, start:] = True
            current_desc = descriptors[-1]
            dist_aff = math.exp(-np.linalg.norm(current_desc[:2]))
            pair_affinity[neighbor_index] = np.clip(0.6 * dist_aff + 0.4 * max(0.0, current_desc[5] + 1.0) * 0.5, 0.0, 1.0)
            pair_label[neighbor_index] = self._pair_label(video_id, frame_id, track_id, neighbor_id)
            pair_mask[neighbor_index] = True

        return {
            "target_feat": self._target_feat(video_id, frame_id, track_id),
            "pair_seq": pair_seq,
            "pair_token_mask": pair_token_mask,
            "pair_affinity": pair_affinity,
            "pair_mask": pair_mask,
            "delta": delta.astype(np.float32),
            "occlusion": np.asarray(self._occlusion_label(video_id, frame_id, track_id), dtype=np.float32),
            "existence": np.asarray(1.0, dtype=np.float32),
            "pair_label": pair_label,
        }


def collate_fn(batch):
    output = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if values[0].dtype == np.bool_:
            output[key] = torch.from_numpy(np.stack(values)).bool()
        else:
            output[key] = torch.from_numpy(np.stack(values)).float()
    return output


def make_parser():
    parser = argparse.ArgumentParser("PGCTrack training")
    parser.add_argument("--ann-file", default="datasets/mot/annotations/train_half.json")
    parser.add_argument("--output-dir", default="outputs/pgctrack")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--memory-len", type=int, default=8)
    parser.add_argument("--max-neighbors", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    args = make_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = PGCMOTDataset(
        args.ann_file,
        memory_len=args.memory_len,
        max_neighbors=args.max_neighbors,
        max_samples=args.max_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )

    model = PGCTrackNet(
        hidden_dim=args.hidden_dim,
        max_len=args.memory_len,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        meters = defaultdict(float)
        progress = tqdm(dataloader, desc="epoch {}/{}".format(epoch, args.epochs))
        for batch in progress:
            batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
            outputs = model(
                batch["target_feat"],
                batch["pair_seq"],
                batch["pair_token_mask"],
                batch["pair_affinity"],
                batch["pair_mask"],
            )
            losses = pgc_loss(outputs, batch)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            for key, value in losses.items():
                meters[key] += float(value.item())
            progress.set_postfix(total="{:.4f}".format(float(losses["total"].item())))

        scheduler.step()
        avg_total = meters["total"] / max(1, len(dataloader))
        ckpt = {
            "model": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "loss": avg_total,
        }
        torch.save(ckpt, os.path.join(args.output_dir, "last.pth"))
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(ckpt, os.path.join(args.output_dir, "best.pth"))
        print("epoch {} loss {:.6f}".format(epoch, avg_total))


if __name__ == "__main__":
    main()
