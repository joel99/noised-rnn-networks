#!/usr/bin/env python3
# Author: Joel Ye

r"""
Notebook for generating density classification task data.

Task is as follows:
Each node is provided an initial binary state 0 or 1 at timestep 0.
The classification of the input is the mode of the input.
The network is then run for T steps, and each node is queried for the classification.
- Being generous, T should be at least network diamter (lnN/lnk for random, or we can probe)
The loss is mean binary cross-entropy.
Task is evaluated on accuracy
TODO what is a sensible baseline?
TODO how do we encourage nodes to reflect correct state at multiple timesteps? (e.g. don't overfit to T, behave like CA)
"""

#%%
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
torch.set_grad_enabled(False)

r"""
See `DensityClassificationDataset` in `src/dataset` for context + usage.
Data Contract:
    task_key
    data: initial states. Shaped B x N
We use N=999 (any subset can be used for classification)
"""

TASK_KEY = "density_classification"
seed = 0
def dc_generator(
    seed=seed,
    n=999,
    num_trials=1000,
    p=0.5,
):
    r"""
        seed: random seed
        n: nodes in network
        num_trials: number of data points
        p: Bernoulli param for IC, or Bernoulli range (e.g. [0,1] as used in Mitchell et al)
    """
    assert n % 2, "n must be odd to guarantee well-defined task density flag"
    torch.manual_seed(seed)
    if isinstance(p, list):
        p_arr = torch.rand(num_trials) * (p[1] - p[0]) + p[0]
        data = p_arr.view(num_trials, 1).expand(num_trials, n) # All nodes within a trial have the same probability
    else:
        data = torch.full((num_trials, n), p)
    data = torch.bernoulli(data)
    return data

#%%
# Generate and save data
num_trials = 10000
n = 999
data = dc_generator(
    n=n,
    num_trials=num_trials,
    p=0.5
)

# Quickly check statistics on labels (should be ~.5)
labels = torch.sum(data, dim=1) > (n / 2)
print(labels.unique(return_counts=True))
train_data, val_data = train_test_split(data, test_size=0.2)

data_dir = "/nethome/jye72/projects/noised-rnn-networks/data"
os.makedirs(data_dir, exist_ok=True)

# Should be at least network diameter
train_dict = dict(
    key=TASK_KEY,
    data=train_data,
)
val_dict = dict(
    key=TASK_KEY,
    data=val_data,
)
torch.save(train_dict, osp.join(data_dir, "dc_train_challenge.pth"))
torch.save(val_dict, osp.join(data_dir, "dc_val_challenge.pth"))

#%%
num_trials = 10000 # 100K doesn't really make a difference.
n = 999
data = dc_generator(
    n=n,
    num_trials=num_trials,
    p=[0.0, 1.0]
)

# Quickly check statistics on labels (should be ~.5)
labels = torch.sum(data, dim=1) > (n / 2)
print(labels.unique(return_counts=True))
train_data, val_data = train_test_split(data, test_size=0.2)

data_dir = "/nethome/jye72/projects/noised-rnn-networks/data"
os.makedirs(data_dir, exist_ok=True)

# Should be at least network diameter
train_dict = dict(
    key=TASK_KEY,
    data=train_data,
)
val_dict = dict(
    key=TASK_KEY,
    data=val_data,
)
print(torch.sum(data, dim=1))

torch.save(train_dict, osp.join(data_dir, "dc_train.pth"))
torch.save(val_dict, osp.join(data_dir, "dc_val.pth"))