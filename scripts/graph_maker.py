#!/usr/bin/env python3
# Author: Joel Ye

# Script to define graphs for training models. Graphs are referenced in the config files to build the model.

#%%
import os
import os.path as osp
import hashlib
import matplotlib.pyplot as plt
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
    "erdos-renyi": nx.erdos_renyi_graph,
    "bar-alb": nx.barabasi_albert_graph
}

def make_and_save_graph(params: dict, experiment_name, generator=graph_generator, show=False):
    param_strs = [f"{k}-{v}" for k, v in params.items()]
    param_str = "_".join(param_strs)
    param_id = hashlib.md5(param_str.encode()).hexdigest()[:5]

    # target_dir = osp.join(output_dir, generator)
    target_dir = osp.join(output_dir, experiment_name)
    os.makedirs(target_dir, exist_ok=True)
    graph_name = osp.join(target_dir, f"{param_id}_{param_str}")
    target_path = f"{graph_name}.edgelist"

    gen_class = GENERATORS[generator]

    num_tries = 0
    success = False
    while not success and num_tries < 5:
        G = gen_class(params.get("n"), params.get("p"), seed=params.get("seed"))
        if len(list(nx.connected_components(G))) == 1:
            success = True
        num_tries += 1
    if not success:
        raise Exception("failed to generate one connected component, raise edge density")

    nx.write_edgelist(G, target_path)
    if show:
        nx.draw(G)
        plt.savefig(f"{graph_name}.pdf")
        plt.clf()

# make_and_save_graph()

# make and save graph and a corresponding config file (from a base config)
# This way we have a 1:1 correspondence for running models

#%%
# Test graphs of varying average degree
experiment_name = "dc_dense"
base_param = {"n": 149, "p": 0.05}
for seed in range(1):
    base_param["seed"] = seed
    make_and_save_graph(base_param, experiment_name, show=True)

#%%
experiment_name = "er_n12"
base_param = {
    "n": 12,
    "p": 0.5
}
for seed in range(5):
    base_param["seed"] = seed
    make_and_save_graph(base_param, experiment_name, show=True)

#%%
experiment_name = "er_n149"
base_param = {
    "n": 149,
    "p": 0.05
}
for seed in range(5):
    base_param["seed"] = seed
    make_and_save_graph(base_param, experiment_name, show=True)


#%%
# Play around with graph
G = nx.read_edgelist('../configs/graphs/sinusoid_samples/2221a_n-10_p-0.6_seed-4.edgelist')
nx.eigenvector_centrality(G)

#%%
G = nx.barabasi_albert_graph(149, 4)
nx.draw(G)
nx.eigenvector_centrality(G)
