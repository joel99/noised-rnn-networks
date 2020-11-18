#!/usr/bin/env python3
# Author: Joel Ye

# List dir and run train for all graphs in directory
from typing import Union, List
import os
import os.path as osp

from run import get_parser, prepare_config

# python run_sweep.py -e ./configs/<variant>.yaml --run-type train

def main():
    parser = get_parser()
    args = parser.parse_args()

    run_exp(**vars(args))

def run_exp(
    exp_config: Union[List[str], str],
    suffix=None, # Bounced
    opts=None,
    **kwargs) -> None:
    config, _ = prepare_config(exp_config, "train", "", opts, suffix=suffix)

    graph_dir = config.MODEL.GRAPH_FILE
    assert osp.isdir(graph_dir)
    graphs = sorted(f for f in os.listdir(config.MODEL.GRAPH_FILE) if f.endswith('.edgelist'))
    for graph in graphs:
        graph_full_path = osp.join(graph_dir, graph)
        variant_fn = osp.split(exp_config)[1]
        variant_name = osp.splitext(variant_fn)[0]
        cmd_str = f"sbatch -x calculon ./scripts/train.sh {variant_name}  MODEL.GRAPH_FILE {graph_full_path}"
        os.system(cmd_str)

if __name__ == "__main__":
    main()

# Eyes on the goal
# you need to test whether the command runs in terminal
# then whether this command runs as expected etc

