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

r"""
See `SinusoidDataset` in `src/dataset` for context + usage.
Data Contract:
    task_key
    data: total signal. Shaped B x N x T
    warmup_period: int for feedin timesteps
    trial_period: int for output timesteps
    We should have warmup_period + trial_period = T.
Currently outputs N=20 (though we can use any subset of N)
"""

TASK_KEY = "sinusoid"

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
        num_trials: number of data points
    """
    torch.manual_seed(seed)
    phase = torch.rand((num_trials, dim, 1))
    speed = torch.rand((num_trials, dim, 1))
    line = speed * torch.arange(trial_length) + phase
    sin_data = torch.sin(line)
    noise = torch.rand_like(sin_data) * sigma
    return sin_data + noise

#%%
# Plot data sample to sanity check
data = sinusoidal_generator(num_trials=2)
print(data.size())
for node_data in data[0]:
    plt.plot(node_data)

#%%
# Generate and save data
n = 1000
train_n = 800
val_n = n - train_n

warmup_period = 3
trial_period = 20
data = sinusoidal_generator(
    dim=20,
    sigma=0.1,
    trial_length=warmup_period + trial_period,
    num_trials=n
)

train_data, val_data = torch.split(data, (train_n, val_n))

data_dir = "/nethome/jye72/projects/noised-rnn-networks/data"
os.makedirs(data_dir, exist_ok=True)

train_dict = dict(
    key=TASK_KEY,
    warmup_period=warmup_period,
    trial_period=trial_period,
    data=train_data
)
val_dict = dict(
    key=TASK_KEY,
    warmup_period=warmup_period,
    trial_period=trial_period,
    data=val_data
)
torch.save(train_dict, osp.join(data_dir, "sin_train.pth"))
torch.save(val_dict, osp.join(data_dir, "sin_val.pth"))