import os
import os.path as osp

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import random
import numpy as np
import torch

from src.runner import Runner
from src.model import EvalRegistry
from run import prepare_config, add_suffix

def init(variant, ckpt="lve", base="", prefix="", graph_file=None, device=None):
    # Initialize model
    # If graph file is specified in config, that will be used
    # If config specifies directory, we'll use `graph_file` for the filename
    # If `graph_file` is None, the (alphabetically) first file will be used

    run_type = "eval"
    exp_config = osp.join("../configs", prefix, f"{variant}.yaml")
    if base != "":
        exp_config = [osp.join("../configs", f"{base}.yaml"), exp_config]
    ckpt_path = f"{variant}.{ckpt}.pth"

    config, ckpt_path = prepare_config(
            exp_config, run_type, ckpt_path, [
                "USE_TENSORBOARD", False,
                "SYSTEM.NUM_GPUS", 1,
            ], suffix=prefix, graph_file=graph_file
        )
    if graph_file is None and osp.isdir(config.MODEL.GRAPH_FILE):
        config.defrost()
        graphs = sorted(f for f in os.listdir(config.MODEL.GRAPH_FILE) if f.endswith('.edgelist'))
        graph = graphs[0] # ! Oh shoot. I messed this up.
        config.MODEL.GRAPH_FILE = osp.join(config.MODEL.GRAPH_FILE, graph)
        graph_id = graph[:5]
        add_suffix(config, graph_id)
        ckpt_dir, ckpt_fn = osp.split(ckpt_path)
        ckpt_path = osp.join(ckpt_dir, graph_id, ckpt_fn)
        # Update relative path
        # Incorporate graph file into this loading. Currently, it will use the default one in the config.
        config.freeze()
    runner = Runner(config)
    runner.logger.clear_filehandlers()
    runner.load_device(device=device)
    return runner, ckpt_path

    # ckpt_dict = runner.load_checkpoint(ckpt_path, map_location="cpu")

    # # TODO Call model setup if needed (device loading)

    # runner.model.load_state_dict(ckpt_dict)
    # runner.model.eval()
    # torch.set_grad_enabled(False)
    # # TODO get data, load it to device, and  as well
    # # data = data.to(runner.device)

    # logger.info("Done.")
    # return runner


def delta(reference, perturbed, pulse_indices, task):
    r"""
        Measure delta in metric given perturbed and reference outputs.
        If score is per-node, then this excludes the perturbed node.
    """
    perturb_out = perturbed["outputs"] # B x T (x N) x H
    ref_out = reference["outputs"]
    mask = reference["masks"]
    targets = reference["targets"]
    if len(ref_out.size()) == 4:
        subset_indices = list(set(range(perturb_out.size(-2))) - set(pulse_indices)) # excluded pulsed node
        perturb_out = perturb_out[..., subset_indices, :]
        ref_out = ref_out[..., subset_indices, :]
        mask = mask[..., subset_indices, :]
        targets = targets[..., subset_indices, :]
    perturbed_score = EvalRegistry.eval_task(task, perturb_out, targets, mask)['primary']
    ref_score = EvalRegistry.eval_task(task, ref_out, targets, mask)['primary']
    return perturbed_score - ref_score

def pulse(
    runner,
    ckpt_path,
    strength=1.0,
    dropout=False,
    trials=5,
    step=5, # jsut a nice safe pick
    node=[0],
):
    r"""
        A pulse experiment will accumulate results over a number of trials

        returns: metrics as well as outputs processed by "measure"

    """
    runner.logger.mute()
    n = runner.config.TASK.NUM_NODES
    h = runner.config.MODEL.HIDDEN_SIZE
    t = runner.config.TASK.NUM_STEPS # ! Note, we need to hadcode steps
    all_metrics = []
    all_info = []
    _, ref = runner.eval(ckpt_path) # Should be deterministic
    for trial in range(trials):
        perturbation = torch.zeros(t, n, h, device=runner.device)
        perturbation[step, node] = torch.rand(h, device=runner.device) * strength
        dropout_mask = None
        if dropout:
            dropout_mask = torch.ones(t, n, h, device=runner.device)
            dropout_mask[step, node] = 0.0
        metrics_perturbed, info_perturbed = runner.eval(ckpt_path, perturb=perturbation, dropout_mask=dropout_mask)
        all_metrics.append(metrics_perturbed)
        all_info.append(delta(ref, info_perturbed, node, runner.config.TASK.KEY))
    return all_metrics, all_info

def reset_random_state(config):
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    torch.random.manual_seed(config.SEED)

