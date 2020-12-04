#!/usr/bin/env python3
# Author: Joel Ye

#%%
import os
import os.path as osp

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

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

from analyze_utils import init, pulse, reset_random_state # , rq_1_1

variant = "dc"
runner, ckpt_path = init(variant)
#%%
import networkx as nx
G = nx.read_edgelist(runner.config.MODEL.GRAPH_FILE)
nx.draw(G, with_labels=True)
# plt.savefig("network.pdf")
print(nx.diameter(G))

#%%

metrics, info = runner.eval(ckpt_path)
inputs, outputs, targets, masks = info


inputs = info["inputs"]
outputs = info["outputs"]
targets = info["targets"]
masks = info["masks"]

# DC

# We are most definitely not learning. Why? Signal is sparse.
# inputs: B x T x N x 1
# We want to see what's happening in all nodes over time (T x N)
def show_trial(info, i=0, node=0):
    inputs = info["inputs"].cpu()
    outputs = info["outputs"].cpu()
    targets = info["targets"].cpu()
    time_range = torch.arange(inputs.size(1)) # 23, 23 10 1
    # node_in = inputs[i, :, node]
    # print(inputs.size())
    # print(inputs[i, 0])
    # node_out = outputs[i, :, node]
    # node_target = targets[i, :, node]
    sns.heatmap(outputs[i].squeeze(), vmax=2.5, vmin=-1.5, cbar_kws={
        'label': 'Logit', 'ticks': np.arange(-1.5, 3.0, 1.0)
    })
    # plt.plot(time_range, node_in, label="input")
    # plt.plot(time_range, node_out, label="prediction")
    # plt.plot(time_range, node_target, label="truth")
    # * (Caption) DC Predictions over Time
    plt.title(f"Target: {int(targets[i, 0, 0].item())}")
    plt.ylabel("$\\leftarrow$ Timestep")
    plt.xlabel("Node $\\rightarrow$", horizontalalignment="left", x=0.0)
    plt.xticks([])
    plt.yticks(np.arange(0, 24, 4), labels=np.arange(0, 24, 4))
    print(np.count_nonzero(outputs[i, -1] > 0.5)) # Decision threshold
    # plt.yticklabels(np.arange(0, 24, 4))
    # plt.title(f"DC Node {node} Input {node_in[0].item()}| Trial {i} Target {node_target[0].item()}")

# show_trial(info, 7, 12) # Positive
# plt.savefig("figures/dc_positive.pdf")
# show_trial(info, 9, 12) # Negative
# plt.savefig("figures/dc_negative.pdf")

#%%
_, t, *_ = masks.size()
n = runner.config.TASK.NUM_NODES
h = runner.config.MODEL.HIDDEN_SIZE
strength = 100.0
perturbation = torch.zeros(t, n, h, device=device)
perturbation_step = 2 # Right in the middle.
nodes_perturbed = np.random.randint(10) # A random set #
dropout=True
dropout_mask = None
if dropout:
    dropout_mask = torch.ones(t, n, h, device=runner.device)
    dropout_mask[perturbation_step, nodes_perturbed] = 0.0
perturbation[perturbation_step, nodes_perturbed] = torch.rand(h, device=device) * strength
metrics_perturbed, info_perturbed = runner.eval(ckpt_path, perturb=perturbation, dropout_mask=dropout_mask)
# show_trial(info_perturbed, 7, 12)
# show_trial(info_perturbed, 9, 12)
# plt.title("Target: 0, Pulse: 100.0")
# plt.savefig("figures/dc_pulse_negative.pdf") # Note there's stochasticity here

# show_trial(info_perturbed, 7, 12)
# plt.title("Target: 0, Pulse: 100.0")
# plt.savefig("figures/dc_pulse_positive.pdf")

show_trial(info_perturbed, 7, 12)
plt.title("Target: 0, Dropout")
plt.savefig("figures/dc_dropout.pdf")

#%%
runner.logger.mute()
metrics_pert, info_pert = pulse(
    runner,
    ckpt_path,
    strength=100.0,
    node=[10],
    dropout=False
)

print(info_pert)

#%%

# RQ1.1: Noise is different than dropout
rq_1_1(runner, ckpt_path)