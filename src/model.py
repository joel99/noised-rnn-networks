#!/usr/bin/env python3
# Author: Joel Ye
import math

import networkx as nx

from torch_geometric.typing import Adj, OptTensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Parameter as Param
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing

from ..inits import uniform

from src import logger

class GraphRNN(MessagePassing):
    r"""
        Adapted from `GatedGraphConv` from `pytorch-geometric`.
        https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gated_graph_conv.html

        Messages are not sent to specific nodes.
        Hidden states will be set to 0, and temporal inputs will be merged in with messages.
        Different from RIMs, we will simply add to aggregate, and there will be no sparse activation.
        Different from GatedGraphConv, we wil accept variable length input and run for a specified number of steps.

        TODO do I need to consider directed vs undirected?

        TODO cite pytorch-geometric
        TODO cite GatedGraphConv
    """
    def __init__(self, config, device):
        super().__init__(aggr=config.aggr)
        self.config = config

        self.hidden_size = config.HIDDEN_SIZE
        self.input_size = config.INPUT_SIZE

        G = nx.read_edgelist(config.GRAPH_FILE) # is this really 2xE
        self.graph = G.edges.values() # edge_index
        # TODO fix above

        if config.INDEPENDENT_DYNAMICS:
            self.rnns = nn.ModuleList([
                nn.GRUCell(config.INPUT_SIZE, self.hidden_size) for _ in range(self.n)
            ])
        else:
            self.rnn = nn.GRUCell(self.hidden_size, self.hidden_size)

        self.state_to_message = nn.Linear(self.hidden_size, self.hidden_size)
        self.mix_input = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)

        self.device = device
        self.reset_parameters()


    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()

    def forward(self, x: Tensor, inputs: Tensor = None) -> Tensor:
        r"""
            Will create messages from state and message neighbors.
            Nodes step recurrent state based on messages and input, if available.
            Args:
                x: node state. B x N x h
                inputs: B x N x input_size or None
            Returns:
                x: node state. B x N x h
        """

        m = self.state_to_message(x)
        # propagate_type: (x: Tensor)
        m = self.propagate(self.graph, x=m) # ! what are propagate parameters? size=None
        if inputs is not None:
            m_and_input = torch.cat([m, inputs], dim=-1)
            m = self.mix_input(m_and_input)
        if self.config.INDEPENDENT_DYNAMICS:
            rnn_states = []
            for i, rnn in enumerate(self.rnns):
                rnn_states.append(rnn(m[:,i], x[:,i]))
            x = torch.stack(dim=1)
        else:
            x = self.rnn(m, x)

        return x

    # ! What are these methods for?
    # TODO make sure these messaging methods are correct
    def message(self, x_j: Tensor):
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,
                                              self.out_channels,
                                              self.num_layers)

"""
It's a bit messy to specify T_input and T_target in a generic way.
Instead, we'll build out separate models for each use case.
"""

class SeqSeqModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.graphrnn = GraphRNN(config, device)

    def forward(self, x, labels, label_mask):
        r"""
            If we only care about certain labels e.g. final timestep, we'll need to drop losses for othehr slots.
            This will just come in the form of a mask, so we can do things in parallel.

            x: B x T x N x H
            labels: B x T x N x H
            label_mask: B x T x N x H. 1 if we should keep the loss, 0 if not.
        """

        # ! Implement!
        if x.size(-1) < self.hidden_size:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)
