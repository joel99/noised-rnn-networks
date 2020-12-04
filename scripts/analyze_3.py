#!/usr/bin/env python3
# Author: Joel Ye

# RQ 3 -- Vs nx statistics
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
import pandas as pd
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

SMALL_SIZE = 12
MEDIUM_SIZE = 15
LARGE_SIZE = 18
palette = sns.color_palette("rocket", 5)
alt_palette = sns.color_palette()
palette[0] = alt_palette[0]

prep_plt()

#%%
# ok, everything appears to be run (and dumped) here already.
def extract(variant: str, perturb="pulse1"):
    all_nodes = []
    all_scores = []
    all_suff = []
    pth = f"{variant}-{perturb}.pth"
    info = torch.load(pth)
    def conv_dict(d):
        clean = {}
        for k in d:
            if isinstance(d[k], int):
                clean[k] = d[k]
            else:
                clean[k] = d[k].item()
        return clean
    sanitized = [conv_dict(d) for d in info]
    df = pd.DataFrame(sanitized)
    # print(len(info))
    # nodes = info['range'].numpy()
    # scores = info['score'].numpy()
    # # all_scores = np.array(all_scores)
    # if "dc" in variant or "seq_mnist" in variant:
    #     all_scores = all_scores * -1
    #     print(all_scores)
    # df = pd.DataFrame([np.array(all_nodes).astype(np.float32), np.array(all_scores).astype(np.float32),])
    # df = df.transpose()
    # df.columns = ['nodes', 'scores']
    return df

variant = "sinusoid"
variant = "seq_mnist"
variant = "dc"
df = extract(f"eval/perturb_all_{variant}", "pulse10")
# df = extract("eval/seq_mnist") # We see cascade here
# df = extract("sinusoid")
df['score'] = df['score'].astype(np.float32)
df['node'] = df['node'].astype(np.uint8)

#%%
# Dummy runner
runner, ckpt_path = init(variant, device=device)
# Create node properties dict
G = nx.read_edgelist(runner.config.MODEL.GRAPH_FILE)
centrality_pairs = {
    'degree': nx.degree_centrality,
    'eig': nx.eigenvector_centrality,
    'close': nx.closeness_centrality,
    'harmonic': nx.harmonic_centrality,
    'between': nx.betweenness_centrality,
    'page': nx.pagerank,
    "katz": nx.katz_centrality_numpy
}
pprint = {
    'degree': 'Degree',
    'eig': 'Eigenvalue',
    'katz': 'Katz',
    'close': 'Closeness',
    'harmonic': 'Harmonic',
    'between': 'Betweenness',
    'page': 'PageRank'
}
# Fix this up...
for metric in centrality_pairs:
    _metric_dict = centrality_pairs[metric](G)
    def apply_dict(node):
        return _metric_dict[str(int(node))]
    # Add node properties
    df[metric] = df.apply(lambda x: apply_dict(x.node), axis=1)

#%%
nx.draw(G)

#%%
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

variant_pprint = {
    'dc': "Density Classification",
    'seq_mnist': "Seq. MNIST",
    "sinusoid": "Sinusoid"
}
from scipy.stats import pearsonr
# from sklearn.metrics import explained_variance_score, r2_score
# from sklearn.linear_model import Ridge
# How well does centrality predict variance in impact of noise (score)??
# decoder = Ridge(alpha=1.0)
# decoder.fit(np.array(df['degree']).reshape(-1, 1), np.array(df['score']).reshape(-1, 1))
# y_pred = decoder.predict(np.array(df['degree']).reshape(-1, 1))
# print(y_pred.shape)
# vaf = explained_variance_score(df['score'] /2, df['score'], multioutput='uniform_average')
# vaf = explained_variance_score(y_pred[:,0], df['score'], multioutput='uniform_average')

corr_info = {metric: pearsonr(df['score'], df[metric])[0] for metric in centrality_pairs}
print(variant)
prep_plt()
for k, v in corr_info.items():
    print(f"{k}:\t {v:.3f}")
centrality = 'eig'
# centrality = 'between'
ax = sns.scatterplot(
    data=df,
    x=centrality,
    # x='degree',
    y='score',
    # y='katz',
    # y='harmonic',
    # y='between',
    # y='eig',
    # df[df['step'] == 5],
    # x="scores",
    # hue='suff',
    # bins=10,
    # multiple="stack",
    # log_scale=True
)
x_lim = ax.get_xlim()
plt.xlim(0, x_lim[1])
print(len(df))
plt.title(f"{variant_pprint[variant]}")
plt.ylabel("Perturb Delta")
plt.xlabel(f"{pprint[centrality]}")
plt.savefig(f"figures/{variant}_centrality.pdf", bbox_inches="tight")
