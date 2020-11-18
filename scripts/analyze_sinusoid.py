#!/usr/bin/env python3
# Author: Joel Ye

# Notebook for interactive model evaluation/analysis
# Allows us to interrogate model on variable data (instead of masked sample again)

#%%
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
import torch.nn.functional as f
from torch.utils import data
import networkx as nx

from analyze_utils import init
from src.model import EvalRegistry

variant = "sinusoid"
variant = "sinusoid_test"
runner, ckpt_path = init(variant)

# We are prototyping on a single seed, and then writing a script to get
# quantitative numbers for all seeds
print(ckpt_path)

#%%
G = nx.read_edgelist(runner.config.MODEL.GRAPH_FILE)
nx.draw(G)
# plt.savefig("network.pdf")
#%%
metrics, info = runner.eval(ckpt_path, save_path=None, log_tb=False, perturb=None)
inputs = info["inputs"]
outputs = info["outputs"]
targets = info["targets"]
masks = info["masks"]

#%%
# Sinusoid
def show_trial(info, i=0, node=0, save_path="sinusoid.pdf"):
    inputs = info["inputs"]
    outputs = info["outputs"]
    targets = info["targets"]
    masks = info["masks"]
    time_range = torch.arange(inputs.size(1)) # 23, 23 10 1
    # node_in = inputs[i, :, node]
    node_out = outputs[i, :, node]
    node_target = targets[i, :, node]
    plt.axvline(3, label="Trial start", color="gray")
    plt.plot(time_range, node_out, label="prediction")
    plt.plot(time_range, node_target, label="truth")
    plt.title(f"Sinusoid Node {node}, Trial {i}")
    plt.legend(loc=(0.7, 0.1))
    plt.savefig(save_path)
show_trial(info, 1, 0)

#%%
_, t, *_ = masks.size()
n = runner.config.TASK.NUM_NODES
h = runner.config.MODEL.HIDDEN_SIZE
def eval_func(info):
    return EvalRegistry.eval_task(
        runner.config.TASK.KEY,
        info["outputs"],
        info["targets"],
        info["masks"]
    )

def delta_mse(reference, perturbed, pulse_indices):
    r"""
        Measure delta in MSE on unperturbed nodes
    """
    targets = reference["targets"]
    error_perturbed = f.mse_loss(perturbed["outputs"], targets, reduction="none")
    error_ref = f.mse_loss(reference["outputs"], targets, reduction="none")
    subset_indices = list(set(range(error_perturbed.size(-2))) - set(pulse_indices)) # excluded pulsed
    subset_perturbed = error_perturbed[..., subset_indices, 0]
    subset_ref = error_ref[..., subset_indices, 0]
    subset_mask = masks[..., subset_indices, 0]

    mse_perturbed = torch.masked_select(subset_perturbed, subset_mask).mean()
    mse_ref = torch.masked_select(subset_ref, subset_mask).mean()
    return mse_perturbed - mse_ref


def pulse(
    strength=1.0,
    trials=5,
    step=n // 2,
    node=[0],
    measure=eval_func
):
    r"""
        A pulse experiment will accumulate results over a number of trials

        returns: metrics as well as outputs processed by "measure"

        # TODO plot effects (rise in MSE) against timestep, strength, node
        # TODO move GPU
    """
    all_metrics = []
    all_info = []
    _, ref = runner.eval(ckpt_path)
    for trial in range(trials):
        perturbation = torch.zeros(t, n, h)
        # Plot against time as well, ideally -- show invariance to node, and invariance against time, weak correlation against strength. (1.0 --> 100.0, 1000 doesn't do any more)
        perturbation[step, node] = torch.rand(h) * strength
        metrics_perturbed, info_perturbed = runner.eval(ckpt_path, perturb=perturbation)
        all_metrics.append(metrics_perturbed)
        all_info.append(delta_mse(ref, info_perturbed, node))
    return all_metrics, all_info

metrics_pert, info_pert = pulse(strength=100.0)
# show_trial(info_perturbed, 1, 0, save_path="sinusoid_perturbed_1x.pdf")

#%%
print(info_pert)

#%%
show_trial(info_perturbed, 1, 10, save_path="sinusoid_perturbed_1x.pdf")
