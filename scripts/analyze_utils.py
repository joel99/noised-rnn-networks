import os
import os.path as osp

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.runner import Runner
from run import prepare_config, add_suffix
def init(variant, ckpt="lve", base="", prefix="", graph_file=None):
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
        graph = graphs[0]
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
    return runner, ckpt_path
    # runner.load_device()

    # ckpt_dict = runner.load_checkpoint(ckpt_path, map_location="cpu")

    # # TODO Call model setup if needed (device loading)

    # runner.model.load_state_dict(ckpt_dict)
    # runner.model.eval()
    # torch.set_grad_enabled(False)
    # # TODO get data, load it to device, and  as well
    # # data = data.to(runner.device)

    # logger.info("Done.")
    # return runner
