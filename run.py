#!/usr/bin/env python3
# Author: Joel Ye

from typing import List, Union, Tuple
import os
import os.path as osp
import shutil
import random

import argparse
import numpy as np
import torch

from yacs.config import CfgNode as CN
from src.config.default import get_config
from src.runner import Runner

DO_PRESERVE_RUNS = False

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type", "-r",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )

    parser.add_argument(
        "--exp-config", "-e",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )

    parser.add_argument(
        "--ckpt-path", "-c",
        default=None,
        type=str,
        help="full path to a ckpt (for eval or resumption)"
    )

    parser.add_argument(
        "--clear-only",
        default=False,
        type=bool,
    )

    parser.add_argument('--sweep', "-s",
        dest='sweep',
        action='store_true',
        help="If true, will run given config with all the graph files in the graph file directory"
    )
    parser.add_argument('--no-sweep', dest='sweep', action='store_false')
    parser.set_defaults(sweep=False)

    parser.add_argument(
        "--suffix",
        default=None,
        type=str,
        help="Override for experiment name"
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    run_exp(**vars(args))

def check_exists(path, preserve=DO_PRESERVE_RUNS):
    if osp.exists(path):
        print(f"{path} exists")
        if not preserve:
            print(f"removing {path}")
            shutil.rmtree(path, ignore_errors=True)
        return True
    return False

def prepare_config(exp_config: Union[List[str], str], run_type: str, ckpt_path="", opts=None, suffix=None) -> Tuple[CN, str]:
    r"""Prepare config node / do some preprocessing

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        ckpt_path: If training, ckpt to resume. If evaluating, ckpt to evaluate.
        opts: list of strings of additional config options.

    Returns:
        Runner, config, ckpt_path
    """
    config = get_config(exp_config, opts)

    # Default behavior is to pull experiment name from config file
    # Bind variant name to directories
    if isinstance(exp_config, str):
        variant_config = exp_config
    else:
        variant_config = exp_config[-1]
    variant_name = osp.split(variant_config)[1].split('.')[0]
    config.defrost()
    config.VARIANT = variant_name
    if suffix is None:
        suffix = config.EXPERIMENT
    add_suffix(config, suffix)

    add_suffix(config, variant_name)

    if osp.exists(config.MODEL.GRAPH_FILE) and not osp.isdir(config.MODEL.GRAPH_FILE):
        graph_id = osp.split(config.MODEL.GRAPH_FILE)[1][:5]
        add_suffix(config, graph_id)

    if ckpt_path is not None:
        if not osp.exists(ckpt_path):
            ckpt_path = osp.join(config.CHECKPOINT_DIR, ckpt_path)

    np.random.seed(config.SEED)
    random.seed(config.SEED)
    torch.random.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True

    return config, ckpt_path

def add_suffix(config: CN, suffix: str):
    config.defrost()
    config.TENSORBOARD_DIR = osp.join(config.TENSORBOARD_DIR, suffix)
    config.CHECKPOINT_DIR = osp.join(config.CHECKPOINT_DIR, suffix)
    config.LOG_DIR = osp.join(config.LOG_DIR, suffix)
    config.freeze()

def run_exp(
    exp_config: Union[List[str], str],
    run_type: str,
    ckpt_path="",
    suffix=None,
    sweep=False,
    opts=None,
    **kwargs) -> None:
    config, ckpt_path = prepare_config(exp_config, run_type, ckpt_path, opts, suffix=suffix)
    if not sweep:
        launch_single(config, run_type, ckpt_path, **kwargs)
    else:
        assert ckpt_path is None, "sweep not supported with checkpoints"
        graph_files = os.listdir(config.MODEL.GRAPH_FILE)
        all_configs = []
        for graph in graph_files:
            graph_cfg = config.clone()
            graph_cfg.defrost()
            graph_cfg.MODEL.GRAPH_FILE = osp.join(config.MODEL.GRAPH_FILE, graph)
            graph_cfg.freeze()
            add_suffix(graph_cfg, graph[:5])
            all_configs.append(graph_cfg)
        for altered_config in all_configs:
            launch_single(altered_config, run_type, ckpt_path, **kwargs)

def launch_single(config: CN, run_type: str, ckpt_path: str, clear_only=False):
    if clear_only:
        check_exists(config.TENSORBOARD_DIR, preserve=False)
        check_exists(config.CHECKPOINT_DIR, preserve=False)
        check_exists(config.LOG_DIR, preserve=False)
        exit(0)
    runner = Runner(config)
    if run_type == "train":
        if ckpt_path is not None:
            runner.train(checkpoint_path=ckpt_path)
        else:
            if DO_PRESERVE_RUNS:
                if check_exists(config.TENSORBOARD_DIR) or \
                    check_exists(config.CHECKPOINT_DIR) or \
                    check_exists(config.LOG_DIR):
                    exit(1)
            else:
                check_exists(config.TENSORBOARD_DIR)
                check_exists(config.CHECKPOINT_DIR)
                check_exists(config.LOG_DIR)
            runner.train()
    elif run_type == "eval":
        runner.eval(checkpoint_path=ckpt_path)

if __name__ == "__main__":
    main()
