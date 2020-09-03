#!/usr/bin/env python3
# Author: Joel Ye

from src.model import (
    GraphRNN
)

MODELS = {
    "GraphRNN": GraphRNN
}

def get_model_class(model_name):
    return MODELS[model_name]