#!/usr/bin/env python3
# Author: Joel Ye
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import logger

class GraphRNN(nn.Module):
    r"""
        TODO
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def forward(self, data, labels):
        return data
