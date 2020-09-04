#!/usr/bin/env python3
# Author: Joel Ye

# Script to define graphs for training models. Graphs are referenced in the config files to build the model.

#%%
import os
import os.path as osp

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import networkx as nx

output_dir = "../configs/graphs/"

graph_generator = "erdos-renyi"
graph_seed = 0
n = 8
p = 0.1

# https://networkx.github.io/documentation/stable/reference/generators.html
GENERATORS = {
    "erdos-renyi": nx.erdos_renyi_graph
}

def make_and_save_graph(generator=graph_generator, seed=graph_seed):
    target_path = osp.join(output_dir, generator, f"{seed}.edgelist")
    gen_class = GENERATORS[generator]

    G = gen_class(n, p, seed=seed)
    nx.write_edgelist(G, target_path)

make_and_save_graph()

# TODO
# make and save graph and a corresponding config file (from a base config)
# This way we have a 1:1 correspondence for running models

#%%
# TODO set up iterator to make many graphs