#!/usr/bin/env python3
# Author: Joel Ye

# Script to define graphs for training models. Graphs are referenced in the config files to build the model.

#%%
import os
import os.path as osp
import hashlib

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import networkx as nx

output_dir = "../configs/graphs/"
os.makedirs(output_dir, exist_ok=True)

graph_generator = "erdos-renyi"
graph_seed = 0
params = []
n = 10
p = 0.6

# https://networkx.github.io/documentation/stable/reference/generators.html
GENERATORS = {
    "erdos-renyi": nx.erdos_renyi_graph
}

def make_and_save_graph(params: dict, experiment_name, generator=graph_generator):
    param_strs = [f"{k}-{v}" for k, v in params.items()]
    param_str = "_".join(param_strs)
    param_id = hashlib.md5(param_str.encode()).hexdigest()[:5]
    # target_dir = osp.join(output_dir, generator)
    target_dir = osp.join(output_dir, experiment_name)
    os.makedirs(target_dir, exist_ok=True)
    target_path = osp.join(target_dir, f"{param_id}_{param_str}.edgelist")

    gen_class = GENERATORS[generator]
    G = gen_class(params.get("n"), params.get("p"), seed=params.get("seed"))
    # TODO disallow >1 CC
    nx.write_edgelist(G, target_path)

# make_and_save_graph()

# make and save graph and a corresponding config file (from a base config)
# This way we have a 1:1 correspondence for running models

#%%
# Test graphs of varying average degree
experiment_name = "sin_seed_test"
base_param = {"n": 10, "p": 0.65}
for seed in range(5):
    base_param["seed"] = seed
    make_and_save_graph(base_param, experiment_name)