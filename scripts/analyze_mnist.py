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

# ! SET YOUR DEVICE HERE
ALLOCATED_DEVICE_ID = 4
os.environ["CUDA_VISIBLE_DEVICES"] = str(ALLOCATED_DEVICE_ID)
import torch

if torch.cuda.device_count() >= 0:
    device = torch.device("cuda", 0)
else:
    device = torch.device("cpu")

import torch.nn.functional as f
from torch.utils import data


from analyze_utils import init

from analyze_utils import init, pulse, reset_random_state #, rq_1_1

variant = "seq_mnist_drop0"
variant = "seq_mnist"

runner, ckpt_path = init(variant, device=device)
#%%
import networkx as nx
G = nx.read_edgelist(runner.config.MODEL.GRAPH_FILE)
nx.draw(G, with_labels=True)
# plt.savefig("network.pdf")
print(nx.diameter(G))

#%%
metrics, info = runner.eval(ckpt_path)
inputs = info["inputs"]
outputs = info["outputs"]
targets = info["targets"]
masks = info["masks"]

#%%
# Viz trial
_, predicted = torch.max(outputs, 2) # B x T
masked_predictions = torch.masked_select(predicted, masks) # B x 1
print(masked_predictions.float().mean())
def show_trial(inputs, predictions, trial=0):
    plt.imshow(inputs[trial, 0].cpu())
    plt.title(f"Pred: {masked_predictions[trial].cpu()} Label: {targets[trial].cpu()}")

show_trial(inputs, masked_predictions, 8)


#%%
# perturbation analysis
_, t = masks.size()
h = runner.config.MODEL.HIDDEN_SIZE
n = runner.config.TASK.NUM_NODES
strength = 10.0
perturbation = torch.zeros(t, n, h, device=device)
dropout_mask = torch.ones(t, n, h, device=device)

perturbation_step = 100 # Right in the middle.
nodes_perturbed = [11] # A random one

# perturbation[perturbation_step, nodes_perturbed] = torch.rand(h, device=device) * strength
# dropout_mask[:, nodes_perturbed] = 0.0
dropout_mask[perturbation_step, nodes_perturbed] = 0.0
# Wat, everything's still doing fine. OH! Because we're getting inputs at each step
metrics, info = runner.eval(ckpt_path, perturb=perturbation, dropout_mask=dropout_mask)

#%%
rq_1_1(runner, ckpt_path)
