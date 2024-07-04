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

from mmdet.distillation.apis.train_distill import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval

from mmdet.distillation import build_distiller

def main():
    dist.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file", default="")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory", default="runs")
    args, opts = parser.parse_known_args()

    cfg = Config.fromfile(args.config)
    # configs.load(args.config, recursive=True)
    # configs.update(opts)

    # cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())
    
    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

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

    distiller_cfg = cfg.get('distiller', None)
    if distiller_cfg is None :
        model = build_model(cfg.model)
        model.init_weights()
        if cfg.get("sync_bn", None):
            if not isinstance(cfg["sync_bn"], dict):
                cfg["sync_bn"] = dict(exclude=[])
            model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    else :
        
        configs.load(cfg.teacher_cfg, recursive=True)
        teacher_cfg = Config(recursive_eval(configs), filename=cfg.teacher_cfg)
        configs.load(cfg.student_cfg, recursive=True)
        student_cfg = Config(recursive_eval(configs), filename=cfg.student_cfg)

        cfg.student_cfg = student_cfg
        cfg.teacher_cfg = teacher_cfg
        
        model = build_distiller(cfg.distiller, teacher_cfg=teacher_cfg, student_cfg=student_cfg,
                                train_cfg=student_cfg.get('train_cfg'),
                                test_cfg=student_cfg.get('test_cfg')).cuda()

    datasets = [build_dataset(student_cfg.data.train)]
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
