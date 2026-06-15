import argparse
import colorsys
import os
import os.path as osp
import time

import cv2
import numpy as np
import torch
from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils import fuse_model, get_model_info, postprocess


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def _tlwh_to_tlbr(tlwh):
    tlwh = np.asarray(tlwh, dtype=float)
    return np.asarray([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]], dtype=float)


def _color_for_id(idx, group_saturation=0.85, group_value=0.95):
    hue = ((idx * 0.61803398875) % 1.0)
    rgb = colorsys.hsv_to_rgb(hue, group_saturation, group_value)
    return tuple(int(c * 255) for c in rgb[::-1])  # BGR


def _blend_color(color, alpha=0.85):
    return tuple(int(c * alpha) for c in color)


def _draw_box(img, tlwh, color, text=None, thickness=2, pred=False):
    x, y, w, h = [int(round(v)) for v in tlwh]
    if pred:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)
    if text:
        ty = max(18, y - 4)
        cv2.putText(img, text, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_trace(img, history, color):
    if len(history) < 2:
        return
    pts = []
    for tlwh in history:
        x, y, w, h = tlwh
        pts.append((int(round(x + 0.5 * w)), int(round(y + 0.5 * h))))
    for p0, p1 in zip(pts[:-1], pts[1:]):
        cv2.line(img, p0, p1, color, 2, cv2.LINE_AA)


class PGCVisualTracker(BYTETracker):
    def __init__(self, args, frame_rate=30):
        super().__init__(args, frame_rate=frame_rate)
        self.track_history = {}
        self.pair_group_cache = {}

    def update(self, output_results, img_info, img_size):
        online_targets = super().update(output_results, img_info, img_size)
        self._refresh_history()
        self._refresh_pair_groups()
        return online_targets

    def _refresh_history(self):
        active_ids = set()
        for track in list(self.tracked_stracks) + list(self.lost_stracks):
            if getattr(track, "track_id", None) is None:
                continue
            active_ids.add(track.track_id)
            self.track_history.setdefault(track.track_id, [])
            self.track_history[track.track_id].append(track.tlwh.copy())
            self.track_history[track.track_id] = self.track_history[track.track_id][-30:]
        # keep history for recently removed ids too
        for tid in list(self.track_history.keys()):
            if tid not in active_ids and len(self.track_history[tid]) > 30:
                self.track_history[tid] = self.track_history[tid][-30:]

    def _refresh_pair_groups(self):
        self.pair_group_cache = {}
        if not getattr(self, "pgc", None):
            return
        group_id = 0
        for key, state in getattr(self.pgc, "pairs", {}).items():
            if state.state not in ("A", "W"):
                continue
            tid_a, tid_b = key
            self.pair_group_cache.setdefault(tid_a, set()).add(group_id)
            self.pair_group_cache.setdefault(tid_b, set()).add(group_id)
            group_id += 1

    def get_frame_groups(self):
        groups = {}
        for track in list(self.tracked_stracks) + list(self.lost_stracks):
            tid = track.track_id
            group_ids = sorted(list(self.pair_group_cache.get(tid, [])))
            groups[tid] = group_ids
        return groups

    def get_group_pairs(self):
        pairs = []
        if not getattr(self, "pgc", None):
            return pairs
        for key, state in self.pgc.pairs.items():
            if state.state not in ("A", "W"):
                continue
            pairs.append((key[0], key[1], state))
        return pairs

    def get_track_pred(self, track):
        return np.asarray(getattr(track, "pgc_pred_tlwh", track.tlwh), dtype=float)


class Predictor(object):
    def __init__(self, model, exp, trt_file=None, decoder=None, device=torch.device("cpu"), fp16=False):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()
            self.model = self.model.half()
        else:
            self.model = self.model.float()

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info


def make_parser():
    parser = argparse.ArgumentParser("PGC pair visualization")
    parser.add_argument("demo", default="video", help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None)
    parser.add_argument("--path", default="./videos/palace.mp4", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam camera id")
    parser.add_argument("--save_result", action="store_true", help="whether to save the result")
    parser.add_argument("-f", "--exp_file", default=None, type=str)
    parser.add_argument("-c", "--ckpt", default=None, type=str)
    parser.add_argument("--device", default="gpu", type=str)
    parser.add_argument("--conf", default=None, type=float)
    parser.add_argument("--nms", default=None, type=float)
    parser.add_argument("--tsize", default=None, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true")
    parser.add_argument("--track_thresh", type=float, default=0.5)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.8)
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6)
    parser.add_argument("--min_box_area", type=float, default=10)
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true")
    parser.add_argument("--use_pgc", dest="use_pgc", default=True, action="store_true")
    parser.add_argument("--no_pgc", dest="use_pgc", action="store_false")
    parser.add_argument("--use_pgc_pair", dest="use_pgc_pair", default=True, action="store_true")
    parser.add_argument("--no_pgc_pair", dest="use_pgc_pair", action="store_false")
    parser.add_argument("--use_pgc_delta", dest="use_pgc_delta", default=True, action="store_true")
    parser.add_argument("--no_pgc_delta", dest="use_pgc_delta", action="store_false")
    parser.add_argument("--pgc_ckpt", type=str, default=None)
    parser.add_argument("--pair_vis_dir", type=str, default=None, help="directory to save per-frame pair overlays")
    parser.add_argument("--pair_min_count", type=int, default=2, help="minimum pair count to visualize a group")
    return parser


def _normalize_frame_size(img_info, img_size):
    img_h, img_w = img_info["height"], img_info["width"]
    scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
    return scale


def _collect_pair_groups(tracker):
    pair_groups = {}
    if not getattr(tracker, "pgc", None):
        return pair_groups

    pairs = tracker.pgc.pairs
    graph = {}
    for (tid_a, tid_b), state in pairs.items():
        if state.state not in ("A", "W"):
            continue
        graph.setdefault(tid_a, set()).add(tid_b)
        graph.setdefault(tid_b, set()).add(tid_a)

    visited = set()
    gid = 0
    for node in graph:
        if node in visited:
            continue
        stack = [node]
        comp = set()
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp.add(u)
            for v in graph.get(u, []):
                if v not in visited:
                    stack.append(v)
        if len(comp) >= 2:
            pair_groups[gid] = sorted(list(comp))
            gid += 1
    return pair_groups


def _assign_group_colors(pair_groups):
    group_colors = {}
    for gid in sorted(pair_groups.keys()):
        group_colors[gid] = _color_for_id(gid)
    return group_colors


def _draw_group_overlay(image, tracker, groups, group_colors):
    canvas = image.copy()
    track_by_id = {}
    for track in list(tracker.tracked_stracks) + list(tracker.lost_stracks):
        track_by_id[track.track_id] = track

    for gid, tids in groups.items():
        color = group_colors[gid]
        for tid in tids:
            track = track_by_id.get(tid)
            if track is None:
                continue
            det_tlwh = np.asarray(track.tlwh, dtype=float)
            pred_tlwh = np.asarray(getattr(track, "pgc_pred_tlwh", det_tlwh), dtype=float)
            hist = tracker.track_history.get(tid, [])
            _draw_trace(canvas, hist, color)
            _draw_box(canvas, det_tlwh, color, text=f"id{tid} g{gid}", thickness=2, pred=False)
            _draw_box(canvas, pred_tlwh, _blend_color(color, 0.65), text=None, thickness=1, pred=True)
            cx, cy = int(det_tlwh[0] + 0.5 * det_tlwh[2]), int(det_tlwh[1] + 0.5 * det_tlwh[3])
            px, py = int(pred_tlwh[0] + 0.5 * pred_tlwh[2]), int(pred_tlwh[1] + 0.5 * pred_tlwh[3])
            cv2.arrowedLine(canvas, (cx, cy), (px, py), color, 2, cv2.LINE_AA, tipLength=0.25)

    header = f"pair groups: {len(groups)}"
    cv2.putText(canvas, header, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    return canvas


def run_image_sequence(predictor, args, exp, save_folder):
    files = get_image_list(args.path) if osp.isdir(args.path) else [args.path]
    files.sort()
    tracker = PGCVisualTracker(args, frame_rate=args.fps)
    results = []
    os.makedirs(save_folder, exist_ok=True)

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path)
        raw_img = img_info["raw_img"]
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info["height"], img_info["width"]], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
        groups = _collect_pair_groups(tracker)
        colors = _assign_group_colors(groups)
        vis_img = _draw_group_overlay(raw_img, tracker, groups, colors)

        if args.save_result:
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), vis_img)

        if frame_id % 20 == 0:
            logger.info(f"Processing frame {frame_id}/{len(files)}")

    if args.save_result:
        with open(osp.join(save_folder, "results.txt"), "w") as f:
            f.writelines(results)


def run_video_sequence(predictor, args, exp, save_folder):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    os.makedirs(save_folder, exist_ok=True)
    save_path = osp.join(save_folder, osp.basename(args.path) if args.demo == "video" else "camera.mp4")
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker = PGCVisualTracker(args, frame_rate=args.fps)
    frame_id = 0
    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            break
        outputs, img_info = predictor.inference(frame)
        if outputs[0] is not None:
            tracker.update(outputs[0], [img_info["height"], img_info["width"]], exp.test_size)
        groups = _collect_pair_groups(tracker)
        colors = _assign_group_colors(groups)
        vis_img = _draw_group_overlay(img_info["raw_img"], tracker, groups, colors)
        vid_writer.write(vis_img)
        if args.pair_vis_dir is not None:
            os.makedirs(args.pair_vis_dir, exist_ok=True)
            cv2.imwrite(osp.join(args.pair_vis_dir, f"{frame_id:06d}.jpg"), vis_img)
        frame_id += 1

    vid_writer.release()
    cap.release()


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    vis_folder = osp.join(output_dir, "pair_vis")
    os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        ckpt_file = args.ckpt if args.ckpt is not None else osp.join(output_dir, "best_ckpt.pth.tar")
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    decoder = None
    trt_file = None
    if args.fp16:
        model = model.half()
    else:
        model = model.float()
    predictor = Predictor(model, exp, trt_file=trt_file, decoder=decoder, device=args.device, fp16=args.fp16)

    save_folder = args.pair_vis_dir or vis_folder
    if args.demo == "image":
        run_image_sequence(predictor, args, exp, save_folder)
    else:
        run_video_sequence(predictor, args, exp, save_folder)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(getattr(args, "opts", []))
    main(exp, args)
