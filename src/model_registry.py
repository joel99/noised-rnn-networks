#!/usr/bin/env python3
# Author: Joel Ye

from src.model import (
    GraphRNN,
    SeqSeqModel
)

MODELS = {
    "GraphRNN": GraphRNN,
    "SeqSeq": SeqSeqModel
}

def get_model_class(model_name):
    return MODELS[model_name]