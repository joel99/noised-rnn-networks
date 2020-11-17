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
# Ignore model logs in notebooks
# logger.mute()

from analyze_utils import init

variant = "sinusoid"
ckpt = 14

# variant = "sinusoid"
# ckpt = 14

runner, ckpt_path = init(variant, ckpt)

#%%
print(ckpt_path)
#%%
G = nx.read_edgelist(runner.config.MODEL.GRAPH_FILE)
nx.draw(G)
plt.savefig("network.pdf")
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
    plt.axvline(3, label="Trial start") # ! This depends on dataset
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
strength = 1.0
perturbation = torch.zeros(t, n, h)
perturbation_step = 10 # Right in the middle.
nodes_perturbed = [0] # A random set
perturbation[perturbation_step, nodes_perturbed] = torch.rand(h) * strength
metrics, info = runner.eval(ckpt_path, save_path=None, log_tb=False, perturb=perturbation)

show_trial(info, 1, 0, save_path="sinusoid_perturbed_1x.pdf")
