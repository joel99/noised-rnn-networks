#!/usr/bin/env python3
# Author: Joel Ye

from typing import List, Optional, Union

from yacs.config import CfgNode as CN

DEFAULT_CONFIG_DIR = "config/"
CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 100

# Name of experiment
_C.VARIANT = "experiment"
_C.USE_TENSORBOARD = True
_C.TENSORBOARD_DIR = "tb/"
_C.CHECKPOINT_DIR = "ckpts/"
_C.LOG_DIR = "logs/"

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------
_C.SYSTEM = CN()
_C.SYSTEM.TORCH_GPU_ID = 0
# Auto-assign if you have free reign to GPUs. False if you're in a managed cluster that assigns GPUs.
_C.SYSTEM.GPU_AUTO_ASSIGN = False
_C.SYSTEM.NUM_GPUS = 1

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
# Each dataset has its defines its own task, which requires the right dataloader as well as the right model.

_C.TASK = CN()
# Task key defines the dataloader to use, and affects the model head.
_C.TASK.KEY = "sinusoid"
_C.TASK.NUM_NODES = 10 # Define task nodes (will modify dataset, and verify compatible model)
_C.TASK.INPUT_SIZE = 1
# For tasks where evaluation happens after some processing time, this defines the evaluation timestep.
# If -1, will default to a task default.
_C.TASK.NUM_STEPS = -1
_C.TASK.AGGREGATE_INPUT = False # Technically these belong to task, but I'm exposing here for convenience
_C.TASK.AGGREGATE_OUTPUT = False

_C.DATA = CN()
_C.DATA.DATAPATH = 'data/'
_C.DATA.TRAIN_FILENAME = 'train.pth'
_C.DATA.VAL_FILENAME = 'val.pth'
_C.DATA.TEST_FILENAME = 'test.pth'
_C.DATA.OVERFIT_TEST = False
_C.DATA.USE_ORACLE = False # In case we need it, add slot for oracle information in dataset

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = "SeqSeq"
_C.MODEL.HIDDEN_SIZE = 32
_C.MODEL.GRAPH_FILE = "data/configs/graphs/"

_C.MODEL.DROPOUT = .1
_C.MODEL.INDEPENDENT_DYNAMICS = False # Do nodes have independent GRU parameters?
_C.MODEL.AGGR = "add" # Message aggregation

# -----------------------------------------------------------------------------
# Train Config
# -----------------------------------------------------------------------------
_C.TRAIN = CN()

_C.TRAIN.DO_VAL = True # Run validation while training
_C.TRAIN.DO_R2 = True # Run validation while training

_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.NUM_UPDATES = 10000 # Max updates (epochs)
_C.TRAIN.MAX_GRAD_NORM = 200.0

_C.TRAIN.LR = CN()
_C.TRAIN.LR.INIT = 1e-2
_C.TRAIN.LR.SCHEDULE = True
_C.TRAIN.LR.RESTARTS = 1
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.EPS = 1e-8 # adam eps
_C.TRAIN.PATIENCE = 500  # early stopping
_C.TRAIN.RAMP_PEEK = True

_C.TRAIN.CHECKPOINT_INTERVAL = 1000
_C.TRAIN.LOG_INTERVAL = 10
_C.TRAIN.VAL_INTERVAL = 10 # Val less often so things run faster


def get_cfg_defaults():
  """Get default LFADS config (yacs config node)."""
  return _C.clone()

def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = get_cfg_defaults()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config

