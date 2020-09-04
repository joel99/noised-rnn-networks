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
_C.SYSTEM.GPU_AUTO_ASSIGN = True # Auto-assign
_C.SYSTEM.NUM_GPUS = 1

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.DATAPATH = 'data/'
_C.DATA.TRAIN_FILENAME = 'train.pth'
_C.DATA.VAL_FILENAME = 'val.pth'
_C.DATA.TEST_FILENAME = 'test.pth'
_C.DATA.OVERFIT_TEST = False

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = "GraphRNN"
_C.MODEL.HIDDEN_SIZE = 16
_C.MODEL.INPUT_SIZE = 4
_C.MODEL.NUM_STEPS = 50 # ! Revisit this. This will entirely depend on the task...
_C.MODEL.GRAPH_FILE = "configs/graphs/"

_C.MODEL.DROPOUT = .1
_C.MODEL.INDEPENDENT_DYNAMICS = False # Do nodes have independent GRU parameters?
_C.MODEL.AGGR = "add" # Message aggregation

# -----------------------------------------------------------------------------
# Train Config
# -----------------------------------------------------------------------------
_C.TRAIN = CN()

_C.TRAIN.DO_VAL = True # Run validation while training
_C.TRAIN.DO_R2 = True # Run validation while training

_C.TRAIN.BATCH_SIZE = 500
_C.TRAIN.NUM_UPDATES = 10000 # Max updates
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
_C.TRAIN.LOG_INTERVAL = 50
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

