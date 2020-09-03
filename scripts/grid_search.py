#!/usr/bin/env python3
# Author: Joel Ye

# Quick little (sampled) grid search script that uses a base yaml
import random
import os
import os.path as osp
from collections import OrderedDict

import json
import yaml
import argparse

from src.config.default import get_config
from src.run import run_exp

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suffix",
        required=True,
        help="Exp suffix",
    )

    parser.add_argument(
        "--base",
        type=str,
        required=True,
        help="path to config yaml containing base config for exp",
    )

    parser.add_argument(
        "--sweep",
        type=str,
        default="configs/sweep.json",
        help="path to sweep config json"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Num sample"
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    run_grid(**vars(args))

def get_sample(hps):
    sample = {}
    for hp_key in hps:
        hp = hps[hp_key]
        if isinstance(hp, dict):
            sample[hp_key] = get_sample(hp)
        else:
            sample[hp_key] = random.sample(hp, 1)[0]
    return sample

def run_grid(suffix: str, base: str, sweep: str, count: int):
    base_cfg_path = base
    with open(sweep, "r") as f:
        hps_dict = json.load(f)
    hps = OrderedDict(hps_dict)
    sampled = []
    while len(sampled) < count:
        cur_samples_hashes = set()
        cur_sample = get_sample(hps)
        cur_hash = json.dumps(cur_sample)
        if cur_hash in cur_samples_hashes:
            continue
        cur_samples_hashes.add(cur_hash)
        sampled.append(cur_sample)

    os.makedirs(f'configs/{suffix}', exist_ok=True)
    for i, sample_cfg in enumerate(sampled):
        print(f"Starting run {i}")
        cfg_filepath = osp.join('configs/', suffix, f'grid_{i}.yaml')
        with open(cfg_filepath, 'w') as f:
            yaml.dump(sample_cfg, f)
        run_exp(exp_config=[base_cfg_path, cfg_filepath], run_type="train", ckpt_path=None, suffix=suffix)

if __name__ == "__main__":
    main()



