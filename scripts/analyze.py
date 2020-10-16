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
# from src import logger
# logger.mute()

from analyze_utils import init

variant = "seq_mnist"
ckpt = 14

variant = "sinusoid"
ckpt

runner, ckpt_path = init(variant, ckpt)
#%%

inputs, outputs, targets, masks = runner.eval(ckpt_path)



#%%
# MNIST
_, predicted = torch.max(outputs, 2) # B x T
masked_predictions = torch.masked_select(predicted, masks) # B x 1
print(masked_predictions.float().mean())
def show_trial(trial=0):
    plt.imshow(inputs[trial, 0])
    plt.title(f"Pred: {masked_predictions[trial]} Label: {targets[trial]}")

show_trial(5)