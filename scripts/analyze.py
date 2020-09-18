#!/usr/bin/env python3
# Author: Joel Ye

# Notebook for interactive model evaluation/analysis
# Allows us to interrogate model on variable data (instead of masked sample again)

#%%
import os
import os.path as osp

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import time
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as f
from torch.utils import data

# Ignore model logs in notebooks
# from src import logger
# logger.mute()

from src.run import prepare_config
from src.runner import Runner

from analyze_utils import init

# TODO
ckpt_path = "/nethome/jye72/share/noised-rnn-networks/sinusoid"