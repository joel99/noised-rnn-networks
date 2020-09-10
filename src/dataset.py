#!/usr/bin/env python3
# Author: Joel Ye
import os.path as osp
import abc

import torch
from torch.utils import data

from src import logger

class TemporalNetworkDataset(data.Dataset):
    r"""
        A temporal network dataset is a modeling problem for N nodes over time.
        It has an input defined for each node, or an aggregate input at each timestep (flagged by `HAS_AGGREGATE_INPUT`).
        Note that if we have aggregate input, typical model behavior will duplicate input for nodes.
            input: B x N x T, or B x T

        It has exactly one of two output targets, each of which come with a mask that defines where and when we measure loss:
        1. An output defined for each node at each timestep.
            node_output: B x N x T
            node_mask: B x N x T
        2. An aggregate output that the network should decode. (e.g. for sequential MNIST)
            net_output: B x T
            net_mask: B x T
    """
    HAS_AGGREGATE_INPUT = False
    HAS_AGGREGATE_TARGET = False
    HAS_NODE_TARGET = True

    def __init__(self, config, filename, mode="train"):
        r"""
            args:
                config: dataset config
                filename: excluding path
                mode: "train" or "val"
        """
        super().__init__()
        logger.info(f"Loading {filename}")
        self.config = config.DATA
        self.datapath = osp.join(self.config.DATAPATH, filename)
        split_path = self.datapath.split(".")
        if split_path[-1] == "pth":
            dataset_dict = torch.load(self.datapath)
            self._initialize_dataset(dataset_dict)
        else:
            raise Exception(f"Unknown dataset extension {split_path[-1]}")

        self.inputs = None
        self.targets = None
        self.masks = None
        if config.DATA.OVERFIT_TEST:
            # Assuming batch dim 0
            self.inputs = self.inputs[:10]
            self.targets = self.targets[:10]
            self.masks = self.masks[:10]

    @abc.abstractmethod
    def _initialize_dataset(self, dataset_dict):
        r"""
            Load inputs, targets, and masks.
            Args:
                dataset_dict: raw payload from dataset
        """
        pass

    def __len__(self):
        r""" Number of samples. """
        return self.inputs.size(0)

    def __getitem__(self, index):
        r"""
            Return: Tuple of
                input: B x T or B x N x T
                target: B x T or B x N x T
                mask: same as output
        """
        return self.inputs[index], self.targets[index], self.mask[index]

    def get_dataset(self):
        return self.inputs, self.targets, self.masks

class SinusoidDataset(TemporalNetworkDataset):
    r"""
        Sinusoidal task definition.
        Each node is responsible for predicting the output of a sinusoid parameterized by a speed and phase.
        Each node is given K=3 steps of a random sinusoid with frequency < 1 Hz.
        The node predicts the sinusoid for T=20 timesteps.
        The Nyquist-Shannon Sampling Theorem from DSP suggests these samples should define a unique sinusoid (that can be constructed).
        TODO confirm ^ with someone who understands signals.

        The error is minimized by a function that appropriately derives the underlying function given the samples.
        The sinusoids can be noised to make the task slightly harder and less-defined.

        See `data/sinusoid.py` for generation.
    """
    def _initialize_dataset(self, raw_data):
        r"""
            Data Contract:
                data: total signal. Shaped B x N x T
                warmup_period: int for feedin timesteps
                trial_period: int for output timesteps
                We should have warmup_period + trial_period = T.
        """
        warmup_period = raw_data['warmup_period']
        trial_period = raw_data['trial_period']
        data = raw_data['data'][:warmup_period + trial_period]

        self.inputs = data.clone()
        self.inputs[..., warmup_period:] = 0
        self.targets = data.clone()
        self.masks = torch.ones_like(self.targets)
        self.masks[..., :warmup_period] = 0