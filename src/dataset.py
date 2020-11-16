#!/usr/bin/env python3
# Author: Joel Ye
import os.path as osp
import abc
import math
import numpy as np
import scipy.misc

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

class TemporalNetworkDataset(data.Dataset):
    r"""
        A temporal network dataset is a modeling problem for N nodes over time.
        It has an input defined for each node, or an aggregate input at each timestep (flagged by `HAS_AGGREGATE_INPUT`).
        Note that if we have aggregate input, typical model behavior will duplicate input for nodes.
            input: B x T x N x h_in, or B x T x h_in

        It has exactly one of two output targets, each of which come with a mask that defines where and when we measure loss:
        1. An output defined for each node at each timestep.
            node_output: B x T x N x h_out=1
            node_mask: B x T x N
        2. An aggregate output that the network should decode. (e.g. for sequential MNIST)
            net_output: B x T x h_out=1
            net_mask: B x T
    """
    HAS_AGGREGATE_INPUT = False
    HAS_AGGREGATE_TARGET = False
    HAS_NODE_TARGET = True

    def __init__(self, config, task_cfg, filename=None, mode="train"):
        r"""
            args:
                config: dataset config
                filename: excluding path
                mode: "train" or "val"
        """
        super().__init__()

        self.inputs = None
        self.targets = None
        self.masks = None
        self.config = config.DATA
        self.mode = mode
        # self.oracle = None # Oracle information

        if len(filename) > 0:
            self.datapath = osp.join(self.config.DATAPATH, filename)
            split_path = self.datapath.split(".")
            if split_path[-1] == "pth":
                dataset_dict = torch.load(self.datapath)
                if "key" not in dataset_dict or dataset_dict["key"] != task_cfg.KEY:
                    raise Exception(f"Unexpected dataset task {dataset_dict['key']}. Expected configured task {task_cfg.KEY}")
                self._initialize_dataset(dataset_dict, task_cfg)
            else:
                raise Exception(f"Unknown dataset extension {split_path[-1]}")

            if config.DATA.OVERFIT_TEST:
                # Assuming batch dim 0
                self.inputs = self.inputs[:10]
                self.targets = self.targets[:10]
                self.masks = self.masks[:10]
                # self.oracle = self.oracle[:10] if self.config.USE_ORACLE else None
        else:
            self.init_without_files(task_cfg)

    @abc.abstractmethod
    def init_without_files(self, task_cfg):
        pass

    @abc.abstractmethod
    def _initialize_dataset(self, dataset_dict, task_cfg):
        r"""
            Load inputs, targets, and masks, (shaped as in `__getitem__`)
            Args:
                dataset_dict: raw payload from dataset
                task_cfg: task spec defining data to load
        """
        pass

    def __len__(self):
        r""" Number of samples. """
        return self.inputs.size(0)

    def __getitem__(self, index):
        r"""
            Return: Tuple of
                input: T x H_in or T x N x H_in
                target: T x H_out or T x N x H_out
                mask: T or T x N
        """
        return (
            self.inputs[index],
            self.targets[index],
            self.masks[index],
            # self.oracle[index] if self.config.USE_ORACLE else None
        )

    def get_dataset(self):
        return (
            self.inputs,
            self.targets,
            self.masks,
            # self.oracle if self.config.USE_ORACLE else None
        )

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
    def _initialize_dataset(self, dataset_dict, task_cfg):
        r"""
            Data Contract:
                data: total signal. Shaped B x N x T
                warmup_period: int for feedin timesteps
                trial_period: int for output timesteps
                We should have warmup_period + trial_period = T.
        """
        warmup_period = dataset_dict['warmup_period']
        trial_period = dataset_dict['trial_period']
        sin_data = dataset_dict['data'][..., :warmup_period + trial_period] # B x N x T
        sin_data = sin_data[:, :task_cfg.NUM_NODES] # B x N x T
        sin_data = sin_data.permute(0, 2, 1).unsqueeze(-1) # Now shaped B x T x N x 1

        self.inputs = sin_data.clone()
        self.inputs[:, warmup_period:] = 0
        self.targets = sin_data.clone()
        self.masks = torch.ones_like(self.targets, dtype=torch.bool)
        self.masks[:, :warmup_period] = 0

        # if self.config.USE_ORACLE:
            # oracle = dataset_dict['clean_data'][..., :warmup_period + trial_period]
            # oracle = oracle[:, :task_cfg.NUM_NODES]
            # self.oracle = oracle.permute(0, 2, 1).unsqueeze(-1)

class DensityClassificationDataset(TemporalNetworkDataset):
    r"""
        Density Classification task definition, modified from http://csc.ucdavis.edu/~evca/Papers/evca-review.pdf
        Each node is given an initial binary state.
        At timestep T, nodes predict the majority initial state (e.g. 1 if more than half of initial states are 1.)

        The network should be able to learn to store a local state and communicate until a global state is determined.

        See `data/density_classification.py` for generation.
    """
    def _initialize_dataset(self, dataset_dict, task_cfg):
        r"""
            Data Contract:
                data: total signal. Shaped B x N
        """
        # Calculate T (depends on diameter of network..)
        initial_states = dataset_dict["data"][:, :task_cfg.NUM_NODES]
        B, N = initial_states.size()
        if task_cfg.NUM_STEPS < 0:
            T = int(3 * math.log(N) / math.log(2)) # For N from 149 to 999, this is around 20 - 30 timesteps.
        else:
            T = task_cfg.NUM_STEPS
        # We require our graphs to be fully connected, for simplicity, so the true diameter is shorter (6-10).
        # So this should be ample computation time?
        self.inputs = torch.zeros((B, T, N, 1))
        self.inputs[:, 0] = initial_states.view(B, N, 1)
        self.targets = (torch.sum(initial_states, dim=1) > (N / 2)).view(B, 1, 1, 1).expand(B, T, N, 1).float()
        self.masks = torch.zeros_like(self.targets, dtype=torch.bool)
        self.masks[:, T-1] = 1 # Only evaluate in final timestep

class SequentialMNISTDataset(TemporalNetworkDataset):
    r"""
        Pixel-wise Sequential MNIST.
        Wraps over RIMs sequential MNIST dataloaders.
        # TODO support val
    """
    HAS_AGGREGATE_INPUT = True
    HAS_AGGREGATE_TARGET = True

    def __init__(self, config, task_cfg, filename=None, mode="train"):
        assert not config.DATA.OVERFIT_TEST, "Unsupported"
        super().__init__(config, task_cfg, filename=filename, mode=mode)

    def init_without_files(self, task_cfg):
        self.dataset = self.get_mnist_dataset()
        if task_cfg.NUM_STEPS < 0:
            T = 196 # Hard-coded, revisit with below upsampling
            # T = 49 # Hard-coded, revisit with below upsampling
        else:
            T = task_cfg.NUM_STEPS
        self.masks = torch.zeros((len(self.dataset), T), dtype=torch.bool)
        self.masks[:, -1] = 1

    def __getitem__(self, index):
        img, label = self.dataset[index] # 1 x w x h
        # For batching efficiency, preprocessing will happen in the model itself
        return (
            img,
            label,
            self.masks[index]
        )

    def __len__(self):
        return len(self.dataset)


    def _initialize_dataset(self, dataset_dict, task_cfg):
        pass

    def get_mnist_dataset(self):
        '''
        Adapted from https://raw.githubusercontent.com/anirudh9119/RIMs/master/event_based/mnist_seq_data_classify.py
        Returns:
            x: (784,50000) int32.
            y: (784,50000) int32.
        '''
        path = self.config.DATAPATH
        # We'll use test as val -- we're not trying to break any benchmarks here, nor are we worreid about overfitting to val
        mnist_dataset = datasets.MNIST(
            root=path,
            train=(self.mode == "train"),
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # https://gist.github.com/kdubovikov/eb2a4c3ecadd5295f68c126542e59f0a
        )
        return mnist_dataset

class DatasetRegistry:
    _registry = {
        "sinusoid": SinusoidDataset,
        "density_classification": DensityClassificationDataset,
        "seq_mnist": SequentialMNISTDataset
    }

    @classmethod
    def get_dataset(cls, key) -> TemporalNetworkDataset:
        if key not in DatasetRegistry._registry:
            raise Exception(f"{key} dataset not found. Supported datasets are {DatasetRegistry._registry.keys()}")
        return DatasetRegistry._registry[key]