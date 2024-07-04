import onnx
import onnx2pytorch
import torch
import argparse
from mmcv import Config
from torchpack.utils.config import configs
from mmcv.runner import build_runner, wrap_fp16_model, load_checkpoint
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval
import time
from mmdet.datasets import build_dataloader
from mmdet3d.apis import single_gpu_test


def parse_args() :

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )

def main() :


    args = parse_args()
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    dataset = build_dataset(cfg.data.test)
    samples_per_gpu = 1
    dataloader = build_dataloader(dataset, samples_per_gpu, 4)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    model = wrap_fp16_model(model)    
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    
    #outputs = single_gpu_test(model, dataloader)

if __name__ == "__main__":
    main()
