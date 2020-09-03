import os
import os.path as osp

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch

from src import logger
from src.run import prepare_config
from src.runner import Runner

def init(variant, ckpt, base="", prefix=""):
    # Initialize model

    run_type = "eval"
    exp_config = osp.join("../configs", prefix, f"{variant}.yaml")
    if base != "":
        exp_config = [osp.join("../configs", f"{base}.yaml"), exp_config]
    ckpt_path = f"{variant}.{ckpt}.pth"

    config, ckpt_path = prepare_config(
            exp_config, run_type, ckpt_path, [
                "USE_TENSORBOARD", False,
                "SYSTEM.NUM_GPUS", 1,
            ], suffix=prefix
        )
    runner = Runner(config)
    runner.load_device()

    ckpt_dict = runner.load_checkpoint(ckpt_path, map_location="cpu")

    # TODO Call model setup if needed (device loading)

    runner.model.load_state_dict(ckpt_dict)
    runner.model.eval()
    torch.set_grad_enabled(False)
    # TODO get data, load it to device, and  as well
    # data = data.to(runner.device)

    logger.info("Done.")
    return runner
