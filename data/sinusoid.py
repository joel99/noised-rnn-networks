#!/usr/bin/env python3
# Author: Joel Ye

r"""
Notebook for generating sinusoidal task data.

Sinusoid task is as follows:
Each node is provided 2 values `p` and `s` at timestep 0.
The label is then a noised `sin(p + t * s)` for t \in T steps.
The task runs for T steps.

The task is evaluated using MSE, averaged over nodes and timesteps.
Optimal nodes predict `sin(p + t * s)` at timestep t.
"""

#%%
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
torch.set_grad_enabled(False)

def sinusoidal_generator(
    seed=0,
    dim=10,
    sigma=0.1,
    trial_length=20,
    num_trials=1000
):
    r"""
        seed: random seed
        dim: number of different sinusoids, corresponding to number of modules in network
        sigma: noise scale
        trial_length: T
        num_trials: number of samples
    """
    raise NotImplementedError()

#%%
# Plot data sample to sanity check
data = sinusoidal_generator(num_trials=1)
raise NotImplementedError()

#%%
# Generate and save data
n = 1000
train_n = 800
val_n = n - train_n

# TODO: Make data



# ===
train_data, val_data = torch.split(data, train_n, val_n)

data_dir = "/home/joel/Documents"
os.makedirs(data_dir, exist_ok=True)

torch.save(train_data, osp.join(data_dir, "train.pth"))
torch.save(val_data, osp.join(data_dir, "val.pth"))