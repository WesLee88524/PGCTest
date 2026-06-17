# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # DanceTrack annotations
        self.train_ann = "dance_train.json"
        self.val_ann = "dance_val.json"
        # DanceTrack typically uses 800x1440 or 1088x1920
        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.random_size = (18, 32)
        self.max_epoch = 80
        self.print_interval = 20
        self.eval_interval = 5
        self.test_conf = 0.1
        self.nmsthre = 0.7
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1
        # Track parameters for DanceTrack (non-uniform motion)
        self.track_thresh = 0.5
        self.track_buffer = 30
        self.min_box_area = 100

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        dataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "dancetrack"),
            json_file=self.train_ann,
            name='',
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), self.seed, shuffle=True
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader = DataLoader(
            self.dataset,
            num_workers=4,
            pin_memory=True,
            batch_sampler=batch_sampler,
            collate_fn=MOTDataset.collate_fn,
        )

        return dataloader

    def get_eval_loader(self, batch_size, is_distributed, test_dev=False):
        from yolox.data import MOTDatasetAggregated, TrainTransform

        valdataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "dancetrack"),
            json_file=self.val_ann,
            name='test',
            img_size=self.test_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1200,
            ),
        )

        val_loader = DataLoader(
            valdataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            collate_fn=MOTDataset.collate_fn,
        )

        return val_loader

    def get_optimizer(self, batch_size):
        if self.warmup_epochs > 0:
            lr = self.basic_lr_per_img * batch_size * (self.warmup_epochs * self.num_nodes * self.num_processes)
        else:
            lr = self.basic_lr_per_img * batch_size

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

        return optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.data import WarmupScheduler, LinearLR

        scheduler = WarmupScheduler(
            LinearLR(lr, iters_per_epoch * self.no_aug_epochs),
            multiplier=1.0,
            dist_comm=self.get_world_size() > 1,
            updates_per_epoch=iters_per_epoch,
        )

        return scheduler

    def pre_process_image(self, img, img_info, img_size):
        """
        Pre-process image for DanceTrack evaluation
        """
        from yolox.data import MOTDataset
        return MOTDataset.pre_process_image(self, img, img_info, img_size)
