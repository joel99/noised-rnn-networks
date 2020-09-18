#!/usr/bin/env python3
# Author: Joel Ye

# Notebook for creating graph definitions

#%%
import os
import os.path as osp

import matplotlib.pyplot as plt
import networkx as nx

import numpy as np

graph_dir = "../configs/graphs/"
os.makedirs(graph_dir, exist_ok=True)

#%%
seed = 0
N = 10
p = 0.6

graph_id = "test"
G = nx.generators.random_graphs.erdos_renyi_graph(N, p, seed)

nx.draw(G)
print(nx.number_connected_components(G)))
path = osp.join(graph_dir, f"n{N}_p{p}_{graph_id}.edgelist")
nx.write_edgelist(G, path)

#%%
seed = 0
N = 149
p = 0.033

graph_id = "dc"
G = nx.generators.random_graphs.erdos_renyi_graph(N, p, seed)
print(nx.number_connected_components(G))
print(nx.diameter(G))
nx.draw(G)

path = osp.join(graph_dir, f"n{N}_p{p}_{graph_id}.edgelist")
nx.write_edgelist(G, path)
