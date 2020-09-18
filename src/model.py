#!/usr/bin/env python3
# Author: Joel Ye

import networkx as nx

from torch_geometric.typing import Adj, OptTensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Parameter as Param
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing

from src import logger, TaskRegistry

class GraphRNN(MessagePassing):
    r"""
        Adapted from `GatedGraphConv` from `pytorch-geometric`.
        https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gated_graph_conv.html

        Messages are not sent to specific nodes.
        Hidden states will be set to 0, and temporal inputs will be merged in with messages.
        Different from RIMs, we will simply add to aggregate, and there will be no sparse activation.
        Different from GatedGraphConv, we wil accept variable length input and run for a specified number of steps.

        TODO do I need to consider directed vs undirected?
        # ! By default, the edge list is read in as one direction, so we interpret as directed messages

    """
    def __init__(self, config, task_cfg, device):
        super().__init__(aggr=config.AGGR)
        self.config = config

        self.hidden_size = config.HIDDEN_SIZE
        self.input_size = task_cfg.INPUT_SIZE

        G = nx.read_edgelist(config.GRAPH_FILE, nodetype=int) # is this really 2xE
        if len(G) != task_cfg.NUM_NODES:
            raise Exception(f"Task nodes {task_cfg.NUM_NODES} and graph nodes {len(G)} don't match.")
        # Note - this is inherently directed
        self.graph = torch.tensor(list(G.edges), dtype=torch.long, device=device).permute(1, 0) # edge_index

        # We may do experiments with fixed dynamics across nodes
        if config.INDEPENDENT_DYNAMICS:
            self.n = len(G)
            self.rnns = nn.ModuleList([
                nn.GRUCell(config.INPUT_SIZE, self.hidden_size) for _ in range(self.n)
            ])
        else:
            self.rnn = nn.GRUCell(self.hidden_size, self.hidden_size)

        self.state_to_message = nn.Linear(self.hidden_size, self.hidden_size)
        self.mix_input = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)

        self.device = device

    def forward(self, inputs: Tensor, state: Tensor,) -> Tensor:
        r"""
            Will create messages from state and message neighbors.
            Nodes step recurrent state based on messages and input, if available.
            Args:
                inputs: B x N x input_size or None
                state: node state. B x N x h
            Returns:
                state: node state. B x N x h
        """
        b, n, _ = state.size()
        m = self.state_to_message(state)
        m = self.propagate(self.graph, x=m)
        if inputs is not None:
            m_and_input = torch.cat([m, inputs], dim=-1)
            m = self.mix_input(m_and_input)
        if self.config.INDEPENDENT_DYNAMICS:
            rnn_states = []
            for i, rnn in enumerate(self.rnns):
                rnn_states.append(rnn(m[:,i], state[:,i]))
            state = torch.stack(rnn_states, dim=1)
        else:
            # B x N x H, B x N x H. Gru cell only supports 1 batch dim,
            state = self.rnn(m.view(-1, self.hidden_size), state.view(-1, self.hidden_size)).view(b, n, -1)
        return state

    # Note - x_j is source, x_i is target
    def message(self, x_j: Tensor):
        return x_j

    # Since adj is from->to, adj_t is to->from, and matmul gets everything going to x_i, which is reduced
    # ! I don't think this gets used
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/message_passing.html?highlight=message_and_aggregate#
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,
                                              self.out_channels,
                                              self.num_layers)

class SeqSeqModel(nn.Module):
    def __init__(self, config, task_cfg, device):
        super().__init__()
        self.key = task_cfg.KEY
        self.hidden_size = config.HIDDEN_SIZE
        self.device = device
        self.duplicate_inputs = False
        self.graphrnn = GraphRNN(config, task_cfg, device)
        self.init_readout()
        # TODO (mnist) add a module for aggregate readin, aggregate readout here

    def init_readout(self):
        if self.key == TaskRegistry.SINUSOID:
            self.readout = nn.Linear(self.hidden_size, 1)
        elif self.key == TaskRegistry.DENSITY_CLASS:
            self.readout = nn.Linear(self.hidden_size, 1)
        elif self.key == TaskRegistry.SEQ_MNIST:
            self.readout = nn.Linear(self.hidden_size * self.graphrnn.n, 1)

    def get_loss(self, outputs, targets):
        if self.key == TaskRegistry.SINUSOID:
            return 0.5 * (outputs - targets).pow(2)
        if self.key == TaskRegistry.DENSITY_CLASS:
            return F.binary_cross_entropy_with_logits(outputs, targets)
        if self.key == TaskRegistry.SEQ_MNIST:
            return F.binary_cross_entropy_with_logits(outputs, targets)

    def forward(self,
        x,
        targets,
        targets_mask,
        input_mask=None,
        log_state=False,
    ):
        r"""
            Calculate loss.
            x: B x T x H_in or B x T x N x H_in
            targets: B x T x N x H_out=1
            targets_mask: B x T x N. 1 to keep the loss, 0 if not.

            # Not supported
            input_mask: B x T x N. Ignore input when x = 1. ! Don't think supporting B is posssible.
        """
        if input_mask is None and x.size(1) != targets.size(1):
            raise Exception(f"Input ({x.size(1)}) and targets ({targets.size(1)}) misaligned.")
        state = torch.zeros(x.size(0), x.size(2), self.hidden_size, device=self.device)
        all_states = [state]
        outputs = []
        for i in range(x.size(1)):
            state = self.graphrnn(x[:, i], state)
            if log_state:
                all_states.append(state)
            output = self.readout(state) # B x N x 1
            outputs.append(output)
        outputs = torch.stack(outputs, 1) # B x T x N x 1
        loss = self.get_loss(outputs, targets)
        loss = torch.masked_select(loss, targets_mask)
        return loss.mean()
