#!/usr/bin/env python3
# Author: Joel Ye
import os.path as osp

import h5py
import numpy as np
import torch
from torch.utils import data

from src import logger

class NetworkDataset(data.Dataset):
    r"""
        Abstract class for loading datasets with N inputs, T*N labels.
        # TODO write abstract contracts
        # TODO need "data" + "labels"
    """

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
            data = dataset_dict["data"]
        else:
            raise Exception(f"Unknown dataset extension {split_path[-1]}")

        self.data = data
        if config.DATA.OVERFIT_TEST:
            self.data = self.data[:10]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        r"""
            Return shaped shaped T x N ?
        """
        return self.data[index]

    def get_dataset(self):
        return self.data