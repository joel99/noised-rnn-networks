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
ALLOCATED_DEVICE_ID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(ALLOCATED_DEVICE_ID)
import torch

if torch.cuda.device_count() >= 0:
    device = torch.device("cuda", 0)
else:
    device = torch.device("cpu")

import torch.nn.functional as f
from torch.utils import data
import networkx as nx

from analyze_utils import init, pulse, reset_random_state# , rq_1_1

variant = "sinusoid_drop0"
runner, ckpt_path = init(variant, device=device)

# We are prototyping on a single seed, and then writing a script to get
# quantitative numbers for all seeds
print(ckpt_path)

#%%
G = nx.read_edgelist(runner.config.MODEL.GRAPH_FILE)
print(runner.config.MODEL.GRAPH_FILE)
nx.draw(G, with_labels=True)
# plt.savefig("network.pdf")
print(nx.diameter(G))
#%%
metrics, info = runner.eval(ckpt_path, save_path=None, log_tb=False, perturb=None)
inputs = info["inputs"]
outputs = info["outputs"]
targets = info["targets"]
masks = info["masks"]

_, t, *_ = masks.size()
n = runner.config.TASK.NUM_NODES
h = runner.config.MODEL.HIDDEN_SIZE

#%%
# Sinusoid
SMALL_SIZE = 12
MEDIUM_SIZE = 15
LARGE_SIZE = 18
def prep_plt():
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    # plt.rc('title', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels

    plt.rc('legend', fontsize=SMALL_SIZE, frameon=False)    # legend fontsize
    plt.style.use('seaborn-muted')
    # plt.figure(figsize=(6,4))
    spine_alpha = 0.5
    plt.gca().spines['right'].set_alpha(0.0)
    plt.gca().spines['bottom'].set_alpha(spine_alpha)
    plt.gca().spines['left'].set_alpha(spine_alpha)
    plt.gca().spines['top'].set_alpha(0.0)

    plt.tight_layout()
prep_plt()
def show_trial(info, i=0, node=0, save_path="sinusoid.pdf"):
    inputs = info["inputs"].cpu()
    outputs = info["outputs"].cpu()
    targets = info["targets"].cpu()
    masks = info["masks"].cpu()
    time_range = torch.arange(inputs.size(1)) # 23, 23 10 1
    # node_in = inputs[i, :, node]
    node_out = outputs[i, :, node]
    node_target = targets[i, :, node]
    plt.axvline(3, label="Trial start", color="gray")
    plt.plot(time_range, node_out, label="Prediction")
    plt.plot(time_range, node_target, label="Truth")
    plt.title(f"Sinusoid Node {node}, Trial {i}")
    plt.legend(frameon=True)
    # plt.legend(loc=(0.8, 0.1))
    plt.savefig(save_path)

# show_trial(info, 1, 0)
# show_trial(info, 1, 1)
# plt.savefig("figures/sinusoid_node.pdf", bbox_inches="tight")

#%%
prep_plt()
_, t, *_ = masks.size()
n = runner.config.TASK.NUM_NODES
h = runner.config.MODEL.HIDDEN_SIZE
strength = 100.0
perturbation = torch.zeros(t, n, h, device=device)
perturbation_step = 2 # Right in the middle.
nodes_perturbed = [0] # A random set
perturbation[perturbation_step, nodes_perturbed] = torch.rand(h, device=device) * strength
metrics_perturbed, info_perturbed = runner.eval(ckpt_path, perturb=perturbation)
# show_trial(info_perturbed, 7, 12)
# show_trial(info_perturbed, 9, 12)
# plt.title("Target: 0, Pulse: 100.0")
# plt.savefig("figures/dc_pulse_negative.pdf") # Note there's stochasticity here

# show_trial(info_perturbed, 1, 0)
# plt.savefig("figures/sinusoid_perturb_node_self.pdf")


show_trial(info_perturbed, 1, 1)
plt.savefig("figures/sinusoid_perturb_node_damp.pdf", bbox_inches="tight")
# plt.savefig("figures/sinusoid_perturb_node.pdf")

