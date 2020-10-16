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
        self.dropout = config.DROPOUT
        self.input_size = task_cfg.INPUT_SIZE

        G = nx.read_edgelist(config.GRAPH_FILE, nodetype=int) # is this really 2xE
        if len(G) != task_cfg.NUM_NODES:
            raise Exception(f"Task nodes {task_cfg.NUM_NODES} and graph nodes {len(G)} don't match.")
        # Note - this is inherently directed
        self.graph = torch.tensor(list(G.edges), dtype=torch.long, device=device).permute(1, 0) # edge_index

        self.n = len(G)
        # We may do experiments with fixed dynamics across nodes
        if config.INDEPENDENT_DYNAMICS:
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
                inputs: B x N x input_size or B x input_size
                state: node state. B x N x h
            Returns:
                state: node state. B x N x h
        """
        b = state.size(0)
        if len(inputs.size()) == 2:
            inputs = inputs.unsqueeze(1).expand(inputs.size(0), self.n, inputs.size(-1))
        else:
            assert state.size(1) == self.n
        m = self.state_to_message(state)
        m = self.propagate(self.graph, x=m)
        if inputs is not None:
            m_and_input = torch.cat([m, inputs], dim=-1)
            m = self.mix_input(m_and_input)
        m = F.dropout(m, p=self.dropout, training=self.training)
        if self.config.INDEPENDENT_DYNAMICS:
            rnn_states = []
            for i, rnn in enumerate(self.rnns):
                rnn_states.append(rnn(m[:,i], state[:,i]))
            state = torch.stack(rnn_states, dim=1)
        else:
            # B x N x H, B x N x H. Gru cell only supports 1 batch dim,
            state = self.rnn(m.view(-1, self.hidden_size), state.view(-1, self.hidden_size)).view(b, self.n, -1)
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

class PooledReadout(nn.Module):
    # Categorical for MNIST
    def __init__(self, hidden_size, n):
        super().__init__()
        self.pool = nn.MaxPool1d(n)
        self.out = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # B x N x H -> B x 1
        return self.out(self.pool(x.permute(0, 2, 1)).squeeze(-1))

class PermutedCE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(*args, **kwargs)
    def forward(self, inputs, targets): # 3-dim input to C in last dim to C in second
        return self.criterion(inputs.permute(0, 2, 1), targets) # N x C x T -> N x T

class SeqSeqModel(nn.Module):
    def __init__(self, config, task_cfg, device):
        super().__init__()
        self.key = task_cfg.KEY
        self.hidden_size = config.HIDDEN_SIZE
        self.device = device
        self.duplicate_inputs = False
        self.graphrnn = GraphRNN(config, task_cfg, device)
        self.init_readout()

    def init_readout(self):
        if self.key == TaskRegistry.SINUSOID:
            self.readout = nn.Linear(self.hidden_size, 1)
            self.criterion = nn.MSELoss(reduction='none')
        elif self.key == TaskRegistry.DENSITY_CLASS:
            self.readout = nn.Linear(self.hidden_size, 1)
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        elif self.key == TaskRegistry.SEQ_MNIST:
            self.readout = PooledReadout(self.hidden_size, self.graphrnn.n)
            self.criterion = PermutedCE(reduction='none')

    def preprocess_inputs(self, x, targets, targets_mask):
        r""" Batched input processing (mainly for MNIST)
            x: B x Raw (e.g 1 x H x W)
            targets: B x Raw (e.g. L)
            targets_mask: B x T, B x T x N
        """
        if self.key == TaskRegistry.SEQ_MNIST:
            img = F.interpolate(x, (7, 7)) # B x 1 x H x W # Super low-res
            seq_img = torch.flatten(img, start_dim=1) # B x T
            seq_label = targets.unsqueeze(1).expand_as(seq_img)
            return seq_img.unsqueeze(-1), seq_label, targets_mask
        else:
            return x, targets, targets_mask

    def forward(self,
        x,
        targets,
        targets_mask,
        input_mask=None,
        log_state=False,
    ):
        r"""
            Calculate loss.
            x: B x T (x N) x H_in
            targets: B x T (x N) x H_out=1
            targets_mask: B x T (x N). 1 to keep the loss, 0 if not.

            # Not supported
            input_mask: B x T x N. Ignore input when x = 1. ! Don't think supporting B is posssible.

            Returns:
                Loss: B' (masked selected) x 1
                Model outputs B x T (x N) x H
        """
        x, targets, targets_mask = self.preprocess_inputs(x, targets, targets_mask)
        if input_mask is None and x.size(1) != targets.size(1):
            raise Exception(f"Input ({x.size(1)}) and targets ({targets.size(1)}) misaligned.")
        state = torch.zeros(x.size(0), self.graphrnn.n, self.hidden_size, device=self.device)
        all_states = [state]
        outputs = []
        for i in range(x.size(1)): # Time
            state = self.graphrnn(x[:, i], state)
            if log_state:
                all_states.append(state)
            output = self.readout(state) # B (x N) x H
            outputs.append(output)
        outputs = torch.stack(outputs, 1) # B x T (x N) x H
        loss = self.criterion(outputs, targets)
        # import pdb
        # pdb.set_trace()
        loss = torch.masked_select(loss, targets_mask)
        return loss, outputs

# Evaluation functions (e.g. accuracy)

def eval_sinusoid(outputs, targets, masks):
    return torch.masked_select(F.mse_loss(outputs, targets), masks)

def eval_dc(outputs, targets, masks):
    r"""
        TODO: confidence?
    """
    predicted = outputs > 0.5
    masked_predictions = torch.masked_select(predicted, masks) # B x 1
    return (masked_predictions == targets).float().mean()


def eval_seq_mnist(outputs, targets, masks):
    _, predicted = torch.max(outputs, 2) # B x T
    masked_predictions = torch.masked_select(predicted, masks) # B x 1
    return (masked_predictions == targets).float().mean()

class EvalRegistry:
    r"""
        Most evals are straightforwardly connected to the loss used.
        The runner will be responsible for weighting the batch metrics based on batch size.
        Each eval has one output per target. This registry wraps the masking step.
        Args:
            outputs: B x T (x N) x H
            targets: B x T (x N) x H
            masks: B x T (x N)
        Returns:
            metric: scalar summary
    """

    _registry = {
        "sinusoid": eval_sinusoid,
        "density_classification": eval_dc,
        "seq_mnist": eval_seq_mnist
    }

    @classmethod
    def eval_task(cls, key: str, outputs: Tensor, targets: Tensor, masks: Tensor) -> float:
        if key not in EvalRegistry._registry:
            raise Exception(f"{key} dataset not found. Supported datasets are {EvalRegistry._registry.keys()}")
        func = EvalRegistry._registry[key]
        return func(outputs, targets, masks)