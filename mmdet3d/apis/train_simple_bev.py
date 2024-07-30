

import torch
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import (
    DistSamplerSeedHook,
    EpochBasedRunner,
    GradientCumulativeFp16OptimizerHook,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
)
from mmdet3d.runner import CustomEpochBasedRunner

from mmdet3d.utils import get_root_logger
from mmdet.core import DistEvalHook
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet3d.datasets.viz_data import compile_data
import numpy as np
def train_model(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
):
    logger = get_root_logger()

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            None,
            dist=distributed,
            seed=cfg.seed,
        )
        for ds in dataset
    ]
    
    # final_dim = (int(224 * 2), int(400 * 2))
    # print('resolution:', final_dim)

    # if True:
    #     resize_lim = [0.8,1.2]
    #     crop_offset = int(final_dim[0]*(1-resize_lim[0]))
    # else:
    #     resize_lim = [1.0,1.0]
    #     crop_offset = 0
    
    # data_aug_conf = {
    #     'crop_offset': crop_offset,
    #     'resize_lim': resize_lim,
    #     'final_dim': final_dim,
    #     'H': 900, 'W': 1600,
    #     'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    #              'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    #     'ncams': 6,
    # }
    # scene_centroid_x = 0.0
    # scene_centroid_y = 1.0 # down 1 meter
    # scene_centroid_z = 0.0

    # scene_centroid_py = np.array([scene_centroid_x,
    #                             scene_centroid_y,
    #                             scene_centroid_z]).reshape([1, 3])
    # scene_centroid = torch.from_numpy(scene_centroid_py).float()

    # XMIN, XMAX = -54, 54
    # ZMIN, ZMAX = -54, 54
    # YMIN, YMAX = -5, 3
    # bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

    # Z, Y, X = 180, 8, 180

    # train_dataloader, val_dataloader = compile_data(
    #     'trainval',
    #     '/home/youngjin/datasets/nuscenes/',
    #     data_aug_conf=data_aug_conf,
    #     centroid=scene_centroid_py,
    #     bounds=bounds,
    #     res_3d=(Z,Y,X),
    #     bsz=4,
    #     nworkers=4,
    #     shuffle=True,
    #     use_radar_filters=False,
    #     seqlen=1, # we do not load a temporal sequence here, but that can work with this dataloader
    #     nsweeps=3,
    #     do_shuffle_cams=True,
    #     get_tids=True,
    # )
    # train_iterloader = iter(train_dataloader)
    # val_iterloader = iter(val_dataloader)

    
    # put model on gpus
    find_unused_parameters = cfg.get("find_unused_parameters", True)
    # Sets the `find_unused_parameters` parameter in
    # torch.nn.parallel.DistributedDataParallel
    # model = MMDistributedDataParallel(
    #     model.cuda(),
    #     device_ids=[torch.cuda.current_device()],
    #     broadcast_buffers=False,
    #     find_unused_parameters=find_unused_parameters,
    # )
    model = model.cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.run_dir,
            logger=logger,
            meta={},
        ),
    )
    
    if hasattr(runner, "set_dataset"):
        runner.set_dataset(dataset)

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp
    distributed = False
    # fp16 setting
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        if "cumulative_iters" in cfg.optimizer_config:
            optimizer_config = GradientCumulativeFp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
        else:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )
    if isinstance(runner, EpochBasedRunner):
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_hook = DistEvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, [("train", 1)])