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
ckpt = 11
graph_file = "n149_"
runner, ckpt_path = init(variant, ckpt, graph_file=graph_file)
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
def show_trial(i=0, node=0):
    time_range = torch.arange(inputs.size(1)) # 23, 23 10 1
    node_in = inputs[i, :, node]
    print(inputs.size())
    print(inputs[i, 0])
    node_out = outputs[i, :, node]
    node_target = targets[i, :, node]
    sns.heatmap(outputs[i].squeeze())
    # plt.plot(time_range, node_in, label="input")
    # plt.plot(time_range, node_out, label="prediction")
    # plt.plot(time_range, node_target, label="truth")
    plt.title(f"DC Node {node} Input {node_in[0].item()}| Trial {i} Target {node_target[0].item()}")
    plt.legend(loc=(0.7, 0.1))
show_trial(9, 12)

#%%
# MNIST
_, predicted = torch.max(outputs, 2) # B x T
masked_predictions = torch.masked_select(predicted, masks) # B x 1
print(masked_predictions.float().mean())
def show_trial(trial=0):
    plt.imshow(inputs[trial, 0])
    plt.title(f"Pred: {masked_predictions[trial]} Label: {targets[trial]}")

show_trial(5)