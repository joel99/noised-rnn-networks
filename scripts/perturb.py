
# RQ 2 - pulse of fixed strength (we run 1 and 10), for every node
# We only record node id, other node properties are pulled during analysis
# (Full perturbation analysis for a checkpoint)
import os
import os.path as osp

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import time
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

if torch.cuda.device_count() >= 0:
    device = torch.device("cuda", 0)
else:
    device = torch.device("cpu")

import torch.nn.functional as f
from torch.utils import data
import networkx as nx
import argparse

from analyze_utils import init, pulse, reset_random_state

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", "-v",
        required=True,
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    perturb_all_nodes(**vars(args))

def perturb_all_nodes(variant):
    runner, ckpt_path = init(variant, device=device)
    experiments = {
        # "dropout": {
        #     "dropout": True,
        # },
        "pulse1": {
            "strength": 1.0
        },
        "pulse10": {
            "strength": 10.0
        },
        # "pulse100": {
        #     "strength": 100.0
        # },
        # "pulse1000": {
        #     "strength": 1000.0
        # },
    }
    for experiment_name in experiments:
        experiment_args = experiments[experiment_name]
        reset_random_state(runner.config)
        # 10 nodes, 10 timesteps, 5 random draws
        node_range = torch.arange(runner.config.TASK.NUM_NODES)
        # node_range = torch.randint(0, runner.config.TASK.NUM_NODES, (settings,))
        step_range = [5, runner.config.TASK.NUM_STEPS // 2, runner.config.TASK.NUM_STEPS - 5] # Early, mid, late
        perturb_exp = []
        print()
        print(f"{runner.config.VARIANT} - {experiment_name}")
        print()
        for step in step_range:
            for node in node_range:
                _, primary_res = pulse(
                    runner,
                    ckpt_path,
                    trials=5,
                    step=step,
                    node=[node],
                    **experiment_args
                )
                perturb_exp.append({
                    "step": step,
                    "node": node,
                    "score": torch.tensor(primary_res).mean()
                })
        torch.save(perturb_exp, f"./eval/perturb_all_{runner.config.VARIANT}-{experiment_name}.pth")

if __name__ == "__main__":
    main()
