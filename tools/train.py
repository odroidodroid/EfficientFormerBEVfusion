import argparse
import copy
import os
import random
import time

import numpy as np
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
import yaml
from mmdet3d.apis.train import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval

# import debugpy
# debugpy.listen(7676)
# print("Wait for debugger...")
# debugpy.wait_for_client()
# print("Debugger attached")

def main():
    dist.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file", default="configs/nuscenes/det/transfusion/pointpillars/lidar/pointpillars.yaml")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory", default="runs")
    parser.add_argument("--prune", type=str, default='none')
    parser.add_argument("--prune_config", type=str, default='mmdet3d/prune/configs/prune_configs/camera/resnet50_prune.yaml')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    if args.prune : 
        configs.load(args.prune_config)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir
    cfg.prune = args.prune
    # dump config
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)]

    model = build_model(cfg.model)
    model.init_weights()
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])


    logger.info(f"Model:\n{model}")
    train_model(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=False,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
