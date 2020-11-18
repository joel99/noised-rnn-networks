#!/usr/bin/env python3
# Author: Joel Ye

import networkx as nx

from typing import Optional
from torch_geometric.typing import Adj, OptTensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Parameter as Param, GRUCell
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_undirected

from src import TaskRegistry

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
    propagate_type = {'x': Tensor, 'edge_embedding': OptTensor}


    def __init__(self, config, task_cfg, device):
        super().__init__(aggr=config.AGGR)
        # self.independent_dynamics = config.INDEPENDENT_DYNAMICS

        self.hidden_size = config.HIDDEN_SIZE
        self.dropout = config.DROPOUT
        self.noise_reg = config.NOISE_REG
        self.input_size = task_cfg.INPUT_SIZE
        self.norm = nn.LayerNorm(self.hidden_size)

        G = nx.read_edgelist(config.GRAPH_FILE, nodetype=int)
        self.n = len(G)
        if self.n != task_cfg.NUM_NODES:
            raise Exception(f"Task nodes {task_cfg.NUM_NODES} and graph nodes {self.n} don't match.")

        edge_index = torch.tensor(list(G.edges), dtype=torch.long, device=device).permute(1, 0) # 2 x E
        self.graph = to_undirected(edge_index)

        self.edge_embedding = None
        if config.EMBED_EDGE:
            self.edge_embedding = nn.Parameter(torch.rand((self.graph.size(-1), self.hidden_size,), device=device))

        # row, col = to_undirected(edge_index)
        # self.graph = SparseTensor(row=row, col=col, sparse_sizes=(self.n, self.n))

        # We may do experiments without fixed dynamics across nodes
        # if self.independent_dynamics:
        #     self.rnns = nn.ModuleList([
        #         nn.GRUCell(self.hidden_size, self.hidden_size) for _ in range(self.n)
        #     ])
        # else:
        self.rnn = nn.GRUCell(self.hidden_size, self.hidden_size)

        self.state_to_message = nn.Linear(self.hidden_size, self.hidden_size)
        self.mix_input = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)

        self.device = device

    def forward(self, inputs: Tensor, state: Tensor) -> Tensor:
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
        m = self.propagate(self.graph, x=m, edge_embedding=self.edge_embedding, size=None)
        if inputs is not None:
            m_and_input = torch.cat([m, inputs], dim=-1)
            m = self.mix_input(m_and_input)
        m = self.norm(m)
        m = F.dropout(m, p=self.dropout, training=self.training)

        # ! No independent dynamics (for JIT)
        # if self.independent_dynamics:
        #     rnn_states = []
        #     for i, rnn in enumerate(self.rnns):
        #         rnn_states.append(rnn(m[:,i], state[:,i]))
        #     state = torch.stack(rnn_states, dim=1)
        # else:

        # B x N x H, B x N x H. Gru cell only supports 1 batch dim,
        state = self.rnn(m.view(-1, self.hidden_size), state.view(-1, self.hidden_size)).view(b, self.n, -1)
        if self.noise_reg > 0:
            state = state + torch.rand_like(state, device=state.device) * self.noise_reg
        return state

    # Note - x_j is source, x_i is target
    def message(self, x_j: Tensor, edge_embedding: OptTensor) -> Tensor:
        # x_j: E x out_channels
        # edge_embedding: E x out_channels -- edge embedding helps nodes recognize what signals are coming from where
        return x_j if edge_embedding is None else x_j + edge_embedding

    # # Since adj is from->to, adj_t is to->from, and matmul gets everything going to x_i, which is reduced
    # # ! This won't get used unless we're using sparse tensors.
    # # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/message_passing.html?highlight=message_and_aggregate#
    # def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
    #     return matmul(adj_t, x, reduce=self.aggr)

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

class GRUWrapper(GRUCell):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = 1

    # Feign being a graph with 1 node
    def forward(self, x, h):
        # x: B x N
        # h: B x 1 x N
        # Out: B x 1 x N
        return super().forward(x, h.squeeze(1)).unsqueeze(1)

class SeqSeqModel(nn.Module):
    def __init__(self, config, task_cfg, device):
        super().__init__()
        self.key = task_cfg.KEY
        self.is_mnist = self.key == TaskRegistry.SEQ_MNIST # for JIT
        self.hidden_size = config.HIDDEN_SIZE
        self.device = device
        self.duplicate_inputs = False
        # if config.TYPE == "GRU":
        #     self.recurrent_network = GRUWrapper(task_cfg.INPUT_SIZE, self.hidden_size)
        # else:
        self.recurrent_network = GraphRNN(config, task_cfg, device).jittable()

        self.init_readout()

    def init_readout(self):
        if self.key == TaskRegistry.SINUSOID:
            self.readout = nn.Linear(self.hidden_size, 1)
            self.criterion = nn.MSELoss(reduction='none')
        elif self.key == TaskRegistry.DENSITY_CLASS:
            self.readout = nn.Linear(self.hidden_size, 1)
            # Hmm... we're not quite getting it.
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        elif self.key == TaskRegistry.SEQ_MNIST:
            self.readout = PooledReadout(self.hidden_size, self.recurrent_network.n)
            self.criterion = PermutedCE(reduction='none')

    def preprocess_inputs(self, x, targets, targets_mask):
        r""" Batched input processing (mainly for MNIST)
            x: B x Raw (e.g 1 x H x W)
            targets: B x Raw (e.g. L)
            targets_mask: B x T, B x T x N
        """
        if self.is_mnist:
            img = F.interpolate(x, (14, 14)) # B x 1 x H x W # Super low-res
            # img = F.interpolate(x, (7, 7)) # B x 1 x H x W # Super low-res
            seq_img = torch.flatten(img, start_dim=1) # B x T
            seq_label = targets.unsqueeze(1).expand_as(seq_img)
            return seq_img.unsqueeze(-1), seq_label, targets_mask
        else:
            return x, targets, targets_mask

    def forward(self,
        x,
        targets,
        targets_mask,
        input_mask: Optional[Tensor] = None,
        log_state: bool = False,
        perturb: Optional[Tensor] = None,
    ):
        r"""
            Calculate loss.
            x: B x T (x N) x H_in
            targets: B x T (x N) x H_out=1
            targets_mask: B x T (x N). 1 to keep the loss, 0 if not.
            perturb: T x N x hidden_size -- additive noise (batch agnostic)

            # Not supported
            input_mask: B x T x N. Ignore input when x = 1. ! Don't think supporting B is posssible.

            Returns:
                Loss: B' (masked selected) x 1
                Model outputs B x T (x N) x H
        """
        x, targets, targets_mask = self.preprocess_inputs(x, targets, targets_mask)
        if input_mask is None and x.size(1) != targets.size(1):
            raise Exception(f"Input ({x.size(1)}) and targets ({targets.size(1)}) misaligned.")
        state = torch.zeros(x.size(0), self.recurrent_network.n, self.hidden_size, device=self.device)
        # all_states = [state]
        outputs = []
        for i in range(x.size(1)): # Time
            state = self.recurrent_network(x[:, i], state) # torchscript doesn't work here.. not sure what the issue is https://discuss.pytorch.org/t/runtimeerror-no-grad-accumulator-for-a-saved-leaf-error/59539/3
            # if log_state:
            #     all_states.append(state)
            output = self.readout(state) # B (x N) x H
            outputs.append(output)

            if perturb is not None:
                state = state + perturb[i].unsqueeze(0).expand_as(state)
        outputs = torch.stack(outputs, 1) # B x T (x N) x H
        loss = self.criterion(outputs, targets)
        # import pdb
        # pdb.set_trace()
        loss = torch.masked_select(loss, targets_mask)
        return loss, outputs

# Evaluation functions (e.g. accuracy)

def eval_sinusoid(outputs, targets, masks):
    return {
        'primary': torch.masked_select(F.mse_loss(outputs, targets, reduction='none'), masks).mean() # mse
    }

def eval_dc(outputs, targets, masks):
    r"""
        outputs: B x T x N x 1
        # ! In cellular automata (mitchell et al) a final convergent state was required to score any points
        # However, this is not a very smooth metric for us to evaluate performance, we will simply take the average
        # Nonetheless we will compute total agreement labels for reference
    """
    outputs = outputs.squeeze(-1)
    targets = targets.squeeze(-1)
    masks = masks.squeeze(-1)
    b, t, n = outputs.size()
    predicted = outputs > 0.5
    consensus_probe = predicted[..., 0:1].expand(b, t, n) # b x t x 1st node
    consensus = consensus_probe.all(-1, keepdim=True).expand_as(masks) # b x t x 1
    consensus_mask = consensus & masks
    targets_matched = predicted == targets # b x t x n
    masked_predictions = torch.masked_select(targets_matched, masks) # B x 1
    consensus_predictions = torch.masked_select(targets_matched, consensus_mask) # <= B x 1
    return {
        'primary': masked_predictions.float().mean(),
        'all_for_one': torch.true_divide(consensus_predictions.sum(), (b*n))
    }
    # ! note because we aggregate over predictions, this is a strict upper bound...
    # We likely have a much lower scorer if we require everything to be the same...
    # return torch.true_divide(masked_predictions.sum(), masked_predictions.size(0))


def eval_seq_mnist(outputs, targets, masks):
    _, predicted = torch.max(outputs, 2) # B x T
    # Masks, in this case is only the last element. So we'll get B x 1
    masked_predictions = torch.masked_select(predicted, masks) # B x 1
    # return (masked_predictions == targets).sum() / masked_predictions.size(0)
    return {
        'primary': (masked_predictions == targets).float().mean()
    }

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
            metric: dict of scalars summary
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