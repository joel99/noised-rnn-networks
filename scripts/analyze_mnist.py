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

variant = "seq_mnist_2"
graph_file = "n10_p"
ckpt = 6

runner, ckpt_path = init(variant, ckpt, graph_file)
#%%

metrics, info = runner.eval(ckpt_path)
inputs = info["inputs"]
outputs = info["outputs"]
targets = info["targets"]
masks = info["masks"]

#%%
# MNIST
_, predicted = torch.max(outputs, 2) # B x T
masked_predictions = torch.masked_select(predicted, masks) # B x 1
print(masked_predictions.float().mean())
def show_trial(trial=0):
    plt.imshow(inputs[trial, 0])
    plt.title(f"Pred: {masked_predictions[trial]} Label: {targets[trial]}")

show_trial(8)


#%%
# perturbation analysis
_, t = masks.size()
h = runner.config.MODEL.HIDDEN_SIZE
n = runner.config.TASK.NUM_NODES
strength = 1.0
perturbation = torch.zeros(t, n, h)

perturbation_step = 25 # Right in the middle.
nodes_perturbed = [1] # A random one
perturbation[perturbation_step, nodes_perturbed] = torch.rand(h) * strength

metrics, info = runner.eval(ckpt_path, save_path=None, log_tb=False, perturb=perturbation)