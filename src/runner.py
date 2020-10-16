#!/usr/bin/env python3
# Author: Joel Ye

import os
import os.path as osp

import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data as torchData
# from pytorch_transformers import AdamW, WarmupCosineWithHardRestartsSchedule

from src import (
    get_model_class,
    TensorboardWriter,
    logger
)

from src.dataset import DatasetRegistry
from src.model import EvalRegistry
from src.utils import linear_decay, get_lightest_gpus

"""
Runner class orchestrates model usage.
TODO: add task based model heads
"""

class Runner:
    def __init__(self, config):
        self.config = config
        self.flush_secs = 10
        self.model = None
        self.aux_tasks = []
        self.aux_task_names = []
        self.optimizer = None
        self.lr_scheduler = None
        self.device = None
        self.device_gpu = None
        self.num_gpus = 0
        if not osp.exists(config.LOG_DIR):
            os.makedirs(config.LOG_DIR, exist_ok=True)
        logfile_path = osp.join(config.LOG_DIR, f"{config.VARIANT}.log")
        logger.add_filehandler(logfile_path)

        self.best_val = {
            "value": 100,
            "update": -1,
        }

    def setup_model(self):
        r"""
            Setup model, assign to device.
        """
        self.load_device()
        self.model = get_model_class(self.config.MODEL.NAME)(self.config.MODEL, self.config.TASK, self.device)
        if self.num_gpus > 1:
            if self.config.SYSTEM.GPU_AUTO_ASSIGN:
                gpu_indices = get_lightest_gpus(self.num_gpus)
            else:
                gpu_indices = list(range(self.num_gpus))
            if self.device_gpu in gpu_indices:
                gpu_indices.remove(self.device_gpu)
            else:
                gpu_indices = gpu_indices[:-1]
            gpu_indices = [self.device_gpu] + gpu_indices # Make sure our primary gpu is first
            self.model = nn.DataParallel(self.model, device_ids=gpu_indices)
        self.model = self.model.to(self.device)

    def _get_parameters(self):
        return self.model.parameters()

    def _do_log(self, update):
        return update > 0 and update % self.config.TRAIN.LOG_INTERVAL == 0

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "optim_state": None if self.optimizer is None else self.optimizer.state_dict(),
            "lr_scheduler": None if self.lr_scheduler is None else self.lr_scheduler.state_dict(),
            "config": self.config,
            "best_val": self.best_val,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
        torch.save(
            checkpoint, osp.join(self.config.CHECKPOINT_DIR, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.
        Assumes model, devices, and other modules are loaded.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing miscellaneous checkpoint info
        """

        ckpt_dict = torch.load(checkpoint_path, *args, **kwargs)
        self.model.load_state_dict(ckpt_dict["state_dict"])
        if "optim_state" in ckpt_dict and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt_dict["optim_state"])
        if "lr_scheduler" in ckpt_dict and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(ckpt_dict["lr_scheduler"])
        if "best_val" in ckpt_dict:
            self.best_val = ckpt_dict["best_val"]
        return ckpt_dict["extra_state"]

    def load_device(self):
        r"""
            Load primary device.
        """
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.num_gpus = min(self.config.SYSTEM.NUM_GPUS, torch.cuda.device_count())
            logger.info(f"Using {self.num_gpus} GPUs")
            gpu_id = self.config.SYSTEM.TORCH_GPU_ID
            if self.config.SYSTEM.GPU_AUTO_ASSIGN:
                gpu_id = get_lightest_gpus(1)[0]
            elif self.num_gpus > 1:
                raise Exception("Can't specify more than one GPU without auto-assign.")
            self.device = (
                torch.device("cuda", gpu_id)
            )
            self.device_gpu = gpu_id

        logger.info(f"Using {self.device}")

    def train(self, checkpoint_path=None) -> None:
        r"""Main method for training model.

        Args:
            checkpoint_path: path of checkpoint to load
        Returns:
            None
        """
        self.setup_model()

        train_cfg = self.config.TRAIN
        task_cfg = self.config.TASK

        dataset_cls = DatasetRegistry.get_dataset(task_cfg.KEY)

        training_set = dataset_cls(self.config, task_cfg, filename=self.config.DATA.TRAIN_FILENAME, mode="train")
        training_generator = torchData.DataLoader(training_set,
            batch_size=train_cfg.BATCH_SIZE, shuffle=True
        )

        if train_cfg.DO_VAL:
            validation_set = dataset_cls(self.config, task_cfg, filename=self.config.DATA.VAL_FILENAME, mode="val")
            validation_generator = torchData.DataLoader(validation_set,
                batch_size=train_cfg.BATCH_SIZE, shuffle=False
            )

        self.optimizer = optim.AdamW(
            list(filter(lambda p: p.requires_grad, self._get_parameters())),
            lr=train_cfg.LR.INIT,
            weight_decay=train_cfg.WEIGHT_DECAY,
            eps=train_cfg.EPS,
        )

        logger.info(
            "number of trainable parameters: {}".format(
                sum(
                    param.numel()
                    for param in self._get_parameters()
                    if param.requires_grad
                )
            )
        )

        if self.optimizer is not None and train_cfg.LR.SCHEDULE:
            # steps_per_update = len(training_set) / self.config.DATA.BATCH_SIZE
            # warmup_updates = 1
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=5)
            # self.lr_scheduler = optim.lr_scheduler.LambdaLR(
            #     self.optimizer,
            #     lambda x: linear_decay(x, train_cfg.NUM_UPDATES)
            #     # warmup_steps=steps_per_update * warmup_updates,
            #     # t_total=steps_per_update * train_cfg.NUM_UPDATES,
            #     # cycles=train_cfg.LR.RESTARTS
            # )

        pth_time = 0
        count_updates = 0
        count_checkpoints = 0

        if checkpoint_path is not None:
            extra_state = self.load_checkpoint(checkpoint_path, map_location="cpu")
            count_updates = extra_state["update"]
            count_checkpoints = extra_state["checkpoint"]
            pth_time = extra_state["pth_time"]

        start_updates = count_updates
        do_stop = False
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(start_updates, train_cfg.NUM_UPDATES):
                # checkpoint model
                if update % train_cfg.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"{self.config.VARIANT}.{count_checkpoints}.pth", dict(
                                update=update,
                                checkpoint=count_checkpoints,
                                pth_time=pth_time,
                            )
                    )
                    count_checkpoints += 1
                t_start = time.time()
                self.model.train()
                for x, targets, masks in training_generator:
                    x = x.to(self.device)
                    targets = targets.to(self.device)
                    masks = masks.to(self.device)
                    loss, _ = self.model(x, targets, masks)
                    loss = loss.mean()

                    if self.optimizer is not None:
                        self.optimizer.zero_grad()
                        loss.backward()
                        params = self._get_parameters()

                        nn.utils.clip_grad_norm_(
                            params, train_cfg.MAX_GRAD_NORM
                        )

                        self.optimizer.step()

                pth_time += time.time() - t_start

                writer.add_scalar(
                    "loss", # train lossget_dataset
                    loss,
                    update,
                )

                # Log stats
                if self._do_log(update):
                    # * We're only logging the loss of the last train step
                    logger.queue_stat("loss", loss.item())

                # Computing val on different interval than log to do early stopping
                if train_cfg.DO_VAL and update % train_cfg.VAL_INTERVAL == 0:
                    self.model.eval()
                    with torch.no_grad():
                        eval_losses = 0
                        total_metric = 0
                        total_count = 0
                        for x, targets, masks in validation_generator:
                            x = x.to(self.device)
                            targets = targets.to(self.device)
                            masks = masks.to(self.device)
                            loss, outputs = self.model(x, targets, masks)
                            batch_size = x.size(0)
                            total_count += batch_size
                            metric = EvalRegistry.eval_task(task_cfg.KEY, outputs, targets, masks)
                            total_metric += batch_size * metric
                            eval_losses += loss.mean().item() * batch_size
                        val_loss = eval_losses / total_count
                        total_metric /= total_count

                        writer.add_scalar(
                            "val_loss",
                            val_loss,
                            update,
                        )

                        writer.add_scalar(
                            "eval_metric", # e.g. accuracy
                            total_metric,
                            update,
                        )

                        if self._do_log(update):
                            logger.queue_stat("val loss", val_loss)

                        if self.best_val["value"] > val_loss:
                            self.best_val["value"] = val_loss
                            self.best_val["update"] = update
                        elif update - self.best_val["update"] > train_cfg.PATIENCE:
                            logger.info(f"Val loss has not improved for {train_cfg.PATIENCE} updates. Stopping...")
                            do_stop = True

                    if self.optimizer is not None and train_cfg.LR.SCHEDULE:
                        self.lr_scheduler.step(val_loss)
                        writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'])

                if self._do_log(update):
                    stat_str = "\t".join([f"{stat[0]}: {stat[1]:.3f}" for stat in logger.empty_queue()])
                    logger.info("update: {}\t{}".format(update, stat_str))
                    logger.info(
                        "update: {}\tpth-time: {:.3f}s\t".format(
                            update, pth_time
                        )
                    )

                if do_stop:
                    return

    def eval(
        self,
        checkpoint_path: str
    ) -> None:
        r"""Evaluates a single checkpoint (in interpretable metrics e.g. accuracy). Logger will print agnostic messages, interpret per task.
        Args:
            checkpoint_path: path of checkpoint

        Returns:
            None
        """
        logger.info(f"Starting evaluation")

        self.setup_model()

        train_cfg = self.config.TRAIN
        task_cfg = self.config.TASK

        dataset_cls = DatasetRegistry.get_dataset(task_cfg.KEY)

        validation_set = dataset_cls(self.config, task_cfg, filename=self.config.DATA.VAL_FILENAME, mode="val")
        validation_generator = torchData.DataLoader(validation_set,
            batch_size=train_cfg.BATCH_SIZE, shuffle=False
        )

        extra_state = self.load_checkpoint(checkpoint_path, map_location="cpu")
        updates = extra_state["update"]

        self.model.eval()
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            with torch.no_grad():
                eval_losses = 0
                total_metric = 0
                total_count = 0
                for x, targets, masks in validation_generator:
                    x = x.to(self.device)
                    targets = targets.to(self.device)
                    masks = masks.to(self.device)
                    loss, outputs = self.model(x, targets, masks)
                    batch_size = x.size(0)
                    total_count += batch_size
                    metric = EvalRegistry.eval_task(task_cfg.KEY, outputs, targets, masks)
                    total_metric += batch_size * metric
                    eval_losses += loss.mean().item() * batch_size
                eval_losses /= total_count
                total_metric /= total_count

                writer.add_scalar(
                    "eval_loss",
                    eval_losses,
                    updates,
                )

                writer.add_scalar(
                    "eval_metric", # e.g. accuracy
                    total_metric,
                    updates,
                )

                logger.info(f"Eval loss: {eval_losses}")
                logger.info(f"Eval metric: {total_metric}")