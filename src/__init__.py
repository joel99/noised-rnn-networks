#!/usr/bin/env python3
# Author: Joel Ye

from src.logger_wrapper import make_logger
from src.tb_wrapper import TensorboardWriter

class TaskRegistry:
    SINUSOID = "sinusoid"
    DENSITY_CLASS = "density_classification"
    SEQ_MNIST = "seq_mnist"

from src.model_registry import (
    get_model_class,
)