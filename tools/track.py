from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluator

import argparse
import os
import random
import warnings
import glob
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path
import shutil


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for dist training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # det args
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--use_pgc", dest="use_pgc", default=True, action="store_true", help="enable pair modeling context.")
    parser.add_argument("--no_pgc", dest="use_pgc", action="store_false", help="disable pair modeling context.")
    parser.add_argument("--use_pgc_pair", dest="use_pgc_pair", default=True, action="store_true", help="enable pair affinity/lifecycle modeling.")
    parser.add_argument("--no_pgc_pair", dest="use_pgc_pair", action="store_false", help="disable pair affinity/lifecycle modeling.")
    parser.add_argument("--use_pgc_delta", dest="use_pgc_delta", default=True, action="store_true", help="enable learned Kalman delta box refinement.")
    parser.add_argument("--no_pgc_delta", dest="use_pgc_delta", action="store_false", help="disable learned Kalman delta box refinement.")
    parser.add_argument("--use_pgc_virtual", dest="use_pgc_virtual", default=False, action="store_true", help="enable virtual maintenance from PGC predictions.")
    parser.add_argument("--allow_pgc_virtual_output", dest="allow_pgc_virtual_output", default=False, action="store_true", help="allow virtual maintained tracks to stay output-visible.")
    parser.add_argument("--no_pgc_low_relax", dest="use_pgc_low_relax", action="store_false", help="disable low confidence association relaxation.")
    parser.add_argument("--no_pgc_pred_assoc", dest="use_pgc_pred_assoc", action="store_false", help="disable association cost computed from PGC-predicted boxes.")
    parser.add_argument("--pgc_ckpt", type=str, default=None, help="checkpoint for the learned PGC delta model.")
    parser.add_argument("--ningbo_debug", dest="use_ningbo_debug", default=False, action="store_false", help="use ningbo debug.")
    return parser


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


def duplicate_mot17_sdp_results(results_folder):
    copied = 0
    for src_path in glob.glob(os.path.join(results_folder, "MOT17-*-SDP.txt")):
        src_name = os.path.splitext(os.path.basename(src_path))[0]
        if not src_name.endswith("-SDP"):
            continue
        base_name = src_name[:-4]
        for suffix in ("FRCNN", "DPM"):
            dst_path = os.path.join(results_folder, "{}-{}.txt".format(base_name, suffix))
            shutil.copyfile(src_path, dst_path)
            copied += 1
    if copied:
        logger.info("Copied {} MOT17 SDP result files to FRCNN/DPM variants.".format(copied))


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    cudnn.benchmark = True

    rank = args.local_rank
    # rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    results_folder = os.path.join(file_name, "track_results")
    os.makedirs(results_folder, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
    if exp.val_ann == "val_half.json":
        exp.eval_keep_mot17_suffixes = ("SDP",)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    #logger.info("Model Structure:\n{}".format(str(model)))

    if args.batch_size != 1:
        logger.info(
            "MOT tracking evaluation is sequential; overriding batch size {} -> 1.".format(
                args.batch_size
            )
        )
    val_loader = exp.get_eval_loader(1, is_distributed, args.test)
    evaluator = MOTEvaluator(
        args=args,
        dataloader=val_loader,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
        )

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    if rank == 0:
        duplicate_mot17_sdp_results(results_folder)

    # start evaluate
    *_, summary = evaluator.evaluate(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder
    )
    logger.info("\n" + summary)

    # evaluate MOTA
    mm.lap.default_solver = 'lap'

    if exp.val_ann == 'val_half.json':
        gt_type = '_val_half'
    else:
        gt_type = ''
    print('gt_type', gt_type)
    if args.mot20:
        gtfiles = glob.glob(os.path.join('datasets/MOT20/train', '*/gt/gt{}.txt'.format(gt_type)))
    else:
        gtfiles = glob.glob(os.path.join('datasets/mot/train', '*/gt/gt{}.txt'.format(gt_type)))
    print('gt_files', gtfiles)
    tsfiles = [f for f in glob.glob(os.path.join(results_folder, '*.txt')) if not os.path.basename(f).startswith('eval')]

    logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logger.info('Loading files.')
    
    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles])    
    
    mh = mm.metrics.create()    
    accs, names = compare_dataframes(gt, ts)
    
    logger.info('Running metrics')
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
               'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
               'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    # summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    # print(mm.io.render_summary(
    #   summary, formatters=mh.formatters, 
    #   namemap=mm.io.motchallenge_metric_names))
    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                       'partially_tracked', 'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logger.info('Completed')


if __name__ == "__main__":
    args = make_parser().parse_args()
    if args.ningbo_debug == True:
        # ==================== 手动覆盖/添加实验配置 ====================
        args.exp_file = "exps/example/mot/yolox_x_mot17_half.py"
        args.ckpt = "pretrained/bytetrack_ablation.pth.tar"  # 通常长参数 -c 对应的属性名是 ckpt，你可以去 parser 确认一下
        args.batch_size = 1                                  # 通常 -b 对应 batch_size
        args.devices = 1                                     # 通常 -d 对应 devices

        # 布尔开关 (Flags)
        args.fp16 = True
        args.fuse = True
        args.use_pgc = True
        args.use_pgc_pair = True
        args.use_pgc_delta = True
        args.no_pgc_pred_assoc = True

        # 实验名称与 PGC 权重路径
        args.experiment_name = "bytetrack_pgc_pair_delta_train"
        args.pgc_ckpt = "YOLOX_outputs/pgc_train_smoke_mot17/latest_pgc_ckpt.pth.tar"
        # =============================================================
    
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )
