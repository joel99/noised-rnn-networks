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

# Ignore model logs in notebooks
# logger.mute()

from analyze_utils import init


variant = "dc"
# graph_file = "n149_"
runner, ckpt_path = init(variant)
#%%

metrics, info = runner.eval(ckpt_path)
inputs, outputs, targets, masks = info


#%%
inputs = info["inputs"]
outputs = info["outputs"]
targets = info["targets"]
masks = info["masks"]


#%%
import seaborn as sns
# DC

# We are most definitely not learning. Why? Signal is sparse.
# inputs: B x T x N x 1
# We want to see what's happening in all nodes over time (T x N)
def show_trial(info, i=0, node=0):
    inputs = info["inputs"]
    outputs = info["outputs"]
    targets = info["targets"]
    time_range = torch.arange(inputs.size(1)) # 23, 23 10 1
    node_in = inputs[i, :, node]
    # print(inputs.size())
    # print(inputs[i, 0])
    node_out = outputs[i, :, node]
    node_target = targets[i, :, node]
    sns.heatmap(outputs[i].squeeze())
    # plt.plot(time_range, node_in, label="input")
    # plt.plot(time_range, node_out, label="prediction")
    # plt.plot(time_range, node_target, label="truth")
    plt.title(f"DC Node {node} Input {node_in[0].item()}| Trial {i} Target {node_target[0].item()}")
show_trial(info, 9, 12)

#%%
_, t, *_ = masks.size()
n = runner.config.TASK.NUM_NODES
h = runner.config.MODEL.HIDDEN_SIZE
strength = 10.0
perturbation = torch.zeros(t, n, h)
perturbation_step = 10 # Right in the middle.
nodes_perturbed = [0] # A random set
perturbation[perturbation_step, nodes_perturbed] = torch.rand(h) * strength
metrics_perturbed, info_perturbed = runner.eval(ckpt_path, save_path=None, log_tb=False, perturb=perturbation)

#%%
show_trial(info_perturbed, 9, 12)
