#!/usr/bin/env python3
# Author: Joel Ye

# RQ 1.1 -- pulse of various strength AND RQ 2 Dropout
# We'll use this infra for Q3 as well (just using the dropout checkpoints)

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
import networkx as nx

from analyze_utils import init, pulse, reset_random_state

def rq_1_1(runner, ckpt_path):
    experiments = {
        "dropout": {
            "dropout": True,
        },
        "pulse1": {
            "strength": 1.0
        },
        "pulse10": {
            "strength": 10.0
        },
        "pulse100": {
            "strength": 100.0
        },
        "pulse1000": {
            "strength": 1000.0
        },
    }
    for experiment_name in experiments:
        experiment_args = experiments[experiment_name]
        reset_random_state(runner.config)
        settings = 10
        # 10 nodes, 10 timesteps, 5 random draws
        node_range = torch.randint(0, runner.config.TASK.NUM_NODES, (settings,))
        step_range = torch.randint(0, runner.config.TASK.NUM_STEPS, (settings,))
        node_scores = []
        for i in range(settings):
            node = node_range[i]
            step = step_range[i]
            _, primary_res = pulse(
                runner,
                ckpt_path,
                trials=5,
                step=step,
                node=[node],
                **experiment_args
            )
            node_scores.append(torch.tensor(primary_res).mean())
        node_scores = torch.tensor(node_scores)
        print(f"{runner.config.VARIANT} - {experiment_name} result: \t{node_scores.mean()}")
        torch.save({ # For safekeeping
            "range": node_range,
            "score": node_scores
        }, f"./eval/{runner.config.VARIANT}-{experiment_name}.pth")

#%%

# variant = "sinusoid"
# variant = "sinusoid_drop0"
# variant = "dc"
# variant = "dc_drop0"
# variant = "seq_mnist"
# variant = "seq_mnist_drop0"

# runner, ckpt_path = init(variant, device=device)
# rq_1_1(runner, ckpt_path)
# _, primary = runner.eval(ckpt_path)
# print(primary.keys())

#%%
# ok, everything appears to be run (and dumped) here already.
import pandas as pd
suffix_map = {
    'dropout': "Dropout",
    'pulse1': "P(1)",
    'pulse10': "P(10)",
    'pulse100': "P(100)",
    'pulse1000': "P(1000)"
}
original_score_ranges = { # lower and upper bounds (MSE will just be percent change)
    'dc_drop0': (0.5, 0.964), # numbers extracted from other eval scripts
    'dc': (0.5, 0.948), # deltas are slightly larger here
    'sinusoid_drop0': (0.114, -1), # deltas are slightly smaller
    'sinusoid': (0.111, -1),
    'seq_mnist_drop0': (0.1, 0.979), # deltas are slightly larger
    'seq_mnist': (0.1, 0.977),
}
def map_score(variant, scores):
    # Map to an interpretable percentage drop in performance
    bounds = original_score_ranges[variant]
    if 'dc' in variant or 'seq_mnist' in variant:
        return scores * -1 / (bounds[1] - bounds[0])
    else:
        return scores / bounds[0]

suffixes = ['dropout', 'pulse1', 'pulse10', 'pulse100', 'pulse1000']
def extract(variant: str):
    all_nodes = []
    all_scores = []
    all_suff = []
    for suffix in suffixes:
        pth = f"{variant}-{suffix}.pth"
        info = torch.load(pth)
        all_nodes.extend(info['range'].numpy())
        all_scores.extend(info['score'].numpy())
        all_suff.extend([suffix_map[suffix]] * len(info['range']))
    all_scores = np.array(all_scores)
    all_scores = map_score(variant[variant.rfind('/')+1:], all_scores)
    all_scores[all_scores < 0] = 0.0001 # Slight buffer for visual
    all_task = [variant[variant.rfind('/') + 1:]] * len(all_scores)
    df = pd.DataFrame([np.array(all_nodes).astype(np.float32), np.array(all_scores).astype(np.float32), all_suff, all_task])
    df = df.transpose()
    df.columns = ['nodes', 'scores', 'suff', 'task']
    df['scores'] = df['scores'].astype(np.float32)
    return df

pprint = {
    "sinusoid": "Sinusoid", #  + Dropout",
    "sinusoid_drop0": "Sinusoid",
    "dc": "DC", #  + Dropout",
    "dc_drop0": "DC",
    "seq_mnist": "Seq. MNIST", #  + Dropout",
    "seq_mnist_drop0": "Seq. MNIST",
}

# TODO get the median shift for each category
# TODO check no map if there is shift

variant = "sinusoid_drop0"
variant ="sinusoid" # Clear cascade here
# Except in the sinusoid task!

variant = "dc_drop0"
variant ="dc" # Note - we floor the changes that are ~0.
# ! Strange, dropout appears to shift perturbation error upwards

# variant = "seq_mnist_drop0"
# variant ="seq_mnist"
# ! Wot, the diffs are bigger when we dropout?

variants = ['dc_drop0', 'sinusoid_drop0', 'seq_mnist_drop0']
# variants = ['dc', 'sinusoid', 'seq_mnist']
df = pd.concat([extract(f"eval/{variant}") for variant in variants])

# # %% Normalize by percentage change here...
SMALL_SIZE = 12
MEDIUM_SIZE = 15
LARGE_SIZE = 18
palette = sns.color_palette("rocket", 5)
alt_palette = sns.color_palette()
palette[0] = alt_palette[0]

#%%
fig = plt.figure()
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
f, axes = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(12, 4))
for i, ax in enumerate(axes):
    sns.histplot(
        df[df['task'] == variants[i]],
        x="scores",
        hue='suff',
        palette=palette,
        bins=10,
        multiple="stack",
        # multiple="dodge",
        log_scale=True,
        legend=False,
        ax=ax
    )
    ax.set_ylabel("Number of Trials")
    ax.set_xlabel("")
    # ax.set_xlabel("% Perf. Drop")

    ax.text(0.5, 0.88, pprint[variants[i]],
         horizontalalignment='center',
         fontsize=20,
         transform = ax.transAxes)
    # ax.set_title(f"{pprint[variants[i]]}", loc=(0.5, 0.9))
    # ax.get_yaxis().set_visible(False)
    # ax.get_xaxis().get_label().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
# ax.set_title('DC')
# plt.title("Pulse Tests")
import matplotlib as mpl
legends = [c for c in axes[0].get_children() if isinstance(c, mpl.legend.Legend)]
handles = axes[0].patches
# legends[0].set_title(f"{pprint[variant]} Perturb")
# print(len(handles[0]))
axes[1].legend(
    # title=f"Perturb",
    # loc="upper center",
    loc=(-1.1, 1.0),
    handles=handles[::-10], # Jeez
    labels=[suffix_map[s] for s in suffixes],
    ncol=5,
    prop={'size': 16}
)
axes[1].set_xlabel("Normalized Performance Drop")
# axes[1].set_xlabel("% Perf. Drop, Trained With Dropout")

plt.savefig(f"figures/no_drop_pulse_hist.pdf", bbox_inches="tight")
# plt.savefig(f"figures/drop_pulse_hist.pdf")
# Welp, will need a way to plot this
# plt.xscale('symlog')

#%%
# Measure shift
variants = ['dc_drop0', 'sinusoid_drop0', 'seq_mnist_drop0']
df = pd.concat([extract(f"eval/{variant}") for variant in variants])
variants = ['dc', 'sinusoid', 'seq_mnist']
df_drop = pd.concat([extract(f"eval/{variant}") for variant in variants])

#%%
# print(df[df['task'] == 'dc_drop0']['scores'])
# need task, diff, suff
def get_diff_series(task):
    # This is the change in relative score from with dropout to without dropout -- we have regressed, it looksl ike
    diff = df_drop[df_drop['task'] == task]['scores'] - df[df['task'] == f'{task}_drop0']['scores']
    suff = df[df['task'] == f'{task}_drop0']['suff']
    task_series = pd.Series([task] * len(suff))
    diff_df = pd.DataFrame([diff, suff, task_series])
    diff_df = diff_df.transpose()
    diff_df.columns = ['scores', 'suff', 'task']
    return diff_df
variants = ['dc', 'sinusoid', 'seq_mnist']
df = pd.concat([get_diff_series(variant) for variant in variants])
#%%

print(df[df['task']=='sinusoid']['scores'])
#%%
prep_plt()

f, axes = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(12, 4))
for i, ax in enumerate(axes):
    sns.histplot(
        df[df['task'] == variants[i]],
        x="scores",
        hue='suff',
        palette=palette,
        bins=10,
        multiple="stack",
        # multiple="dodge",
        log_scale=False,
        legend=False,
        ax=ax
    )
    ax.set_ylabel("Number of Trials")
    ax.set_xlabel("")
    ax.text(0.5, 0.88, pprint[variants[i]],
         horizontalalignment='center',
         fontsize=20,
         transform = ax.transAxes)
    # ax.set_title(f"{pprint[variants[i]]}", loc=(0.5, 0.9))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
# ax.set_title('DC')
# plt.title("Pulse Tests")
import matplotlib as mpl
legends = [c for c in axes[0].get_children() if isinstance(c, mpl.legend.Legend)]
handles = axes[0].patches
# legends[0].set_title(f"{pprint[variant]} Perturb")
# print(len(handles[0]))
axes[1].legend(
    # title=f"Perturb",
    # loc="upper center",
    loc=(-1.1, 1.0),
    handles=handles[::-10], # Jeez
    labels=[suffix_map[s] for s in suffixes],
    ncol=5,
    prop={'size': 16}
)
axes[1].set_xlabel("Difference in Perf. Drop when Adding Dropout")

# ax = sns.histplot(
#     diff_df, x='scores_diff', hue='suff', multiple="stack"
# )
# plt.xscale('symlog')
plt.savefig(f"figures/drop_delta_pulse_hist.pdf", bbox_inches="tight")
