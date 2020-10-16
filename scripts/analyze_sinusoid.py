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

variant = "seq_mnist"
ckpt = 14

variant = "sinusoid"
ckpt = 14

variant = "dc"
ckpt = 1
runner, ckpt_path = init(variant, ckpt)
#%%

inputs, outputs, targets, masks = runner.eval(ckpt_path)


#%%
# Sinusoid
def show_trial(i=0, node=0):
    print(masks[i].size())
    time_range = torch.arange(inputs.size(1)) # 23, 23 10 1
    # node_in = inputs[i, :, node]
    node_out = outputs[i, :, node]
    node_target = targets[i, :, node]
    plt.axvline(3, label="Trial start") # ! This depends on dataset
    plt.plot(time_range, node_out, label="prediction")
    plt.plot(time_range, node_target, label="truth")
    plt.title(f"Sinusoid Eval Node {node}, Trial {i}")
    plt.legend(loc=(0.7, 0.1))
show_trial(5, 6)
