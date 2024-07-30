from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F
import time
from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS
from mmcv.runner import _load_checkpoint

from .base import Base3DFusionModel
import numpy as np
import matplotlib.pyplot as plt
import os
__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]) if encoders["camera"]["vtransform"] is not None else None,
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
                self.encoders["lidar"] = nn.ModuleDict(
                    {
                        "voxelize": voxelize_module,
                        "backbone": build_backbone(encoders["lidar"]["backbone"]),
                    }
                )
                self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)
                self.voxelize_flag = True
            else:
                #voxelize_module = DynamicPillarVFE(**encoders["lidar"]["voxelize"])
                self.encoders["lidar"] = nn.ModuleDict({
                    "backbone": build_backbone(encoders["lidar"]["backbone"])
                    })
                self.voxelize_flag = False
            if encoders["lidar"].get("neck") is not None :
                self.encoders["lidar"].add_module("neck", build_neck(encoders["lidar"]["neck"]))
                self.lidar_neck = True
            else : 
                self.lidar_neck = None
        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None
        if decoder.get("backbone") is not None :
            self.decoder = nn.ModuleDict(
                {
                    "backbone": build_backbone(decoder["backbone"]),
                }
            )
        if decoder.get("neck") is not None :
            self.decoder.add_module("neck", build_neck(decoder["neck"]))
            self.decoder_neck_flag = True
        else :
            self.decoder_neck_flag = None
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            pass
            # self.encoders["camera"]["backbone"].init_weights()
            # pretrained = self.encoders["camera"]["backbone"].init_cfg.checkpoint
            # self.load_ckpt_internimage(pretrained)
        else :
            train_dsvt = False
            if train_dsvt :
                pretrained_dsvt = "pretrained/DSVT_Nuscenes_val.pth"
                self.load_ckpt_dsvt(pretrained_dsvt)
                
    def load_ckpt_internimage(self, pretrained) : 
        state_dict = _load_checkpoint(pretrained, map_location='cpu')
        if 'state_dict' in state_dict :
            _state_dict = state_dict["state_dict"]
        elif 'model' in state_dict :
            _state_dict = state_dict["model"]
        else :
            _state_dict = state_dict
        new_dict = {}
        for key in _state_dict.keys() :
            new_dict['encoders.camera.backbone.'+key] = _state_dict[key]
        self.load_state_dict(new_dict, False)
        
        
    def load_ckpt_dsvt(self, pretrained) :
        state_dict = torch.load(pretrained, map_location='cpu')
        state_dict = state_dict["model_state"]
        new_dict = {}
        for key in state_dict.keys() :
            string = key.split('.')
            if string[0] == 'vfe' :
                new_dict['encoders.lidar.backbone.'+ key] = state_dict[key]
            elif string[0] == 'backbone_3d' :
                latter = ".".join(string[1:])
                new_dict['encoders.lidar.backbone.'+ latter] = state_dict[key]
            elif string[0] == 'backbone_2d' :
                latter = ".".join(string[1:])
                new_dict['decoder.backbone.'+ latter] = state_dict[key]
            elif string[0] == 'dense_head' :
                if string[1] == 'prediction_head' :
                    latter = ".".join(string[2:])
                    new_dict['heads.object.prediction_heads.'+latter] = state_dict[key]
                else :
                    latter = ".".join(string[1:])
                    new_dict['heads.object.'+ latter] = state_dict[key]
        self.load_state_dict(new_dict, False)


    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,**kwargs
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        img = x
        x = x.view(B * N, C, H, W)
        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)
        if not isinstance(x, torch.Tensor):
            x = x[0]

        # return x
    
        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)
        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,**kwargs
        )
        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        #feats, coords, sizes = self.voxelize(x)
        #batch_size = coords[-1, 0] + 1
        if self.voxelize_flag :
            feats, coords, sizes = self.voxelize(x)
            batch_size = coords[-1, 0] + 1
            x = self.encoders["lidar"]["backbone"](feats, coords, batch_size)
        else :
            batch_size = len(x)
            x = self.encoders["lidar"]["backbone"](x, batch_size)
        if self.lidar_neck is not None :
            x = self.encoders["lidar"]["neck"](x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()
        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            if not isinstance(img, torch.Tensor) :
                img = img.data[0].half().cuda()
                points = [points.data[0][0].cuda(), points.data[0][1].cuda(), points.data[0][2].cuda(), points.data[0][3].cuda()]
                camera2ego = camera2ego.data[0].cuda()
                lidar2ego = lidar2ego.data[0].cuda()
                lidar2camera = lidar2camera.data[0].cuda()
                lidar2image = lidar2image.data[0].cuda()
                camera_intrinsics = camera_intrinsics.data[0].cuda()
                camera2lidar = camera2lidar.data[0].cuda()
                img_aug_matrix = img_aug_matrix.data[0].cuda()
                lidar_aug_matrix = lidar_aug_matrix.data[0].cuda()
                metas = metas.data[0]
                gt_masks_bev = gt_masks_bev.cuda()
                gt_bboxes_3d = [gt_bboxes_3d.data[0][0], gt_bboxes_3d.data[0][1], gt_bboxes_3d.data[0][2], gt_bboxes_3d.data[0][3]]
                gt_labels_3d = [gt_labels_3d.data[0][0].cuda(), gt_labels_3d.data[0][1].cuda(), gt_labels_3d.data[0][2].cuda(), gt_labels_3d.data[0][3].cuda()]
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,**kwargs
                )
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)
            visualize_bev = False
            if visualize_bev :
                bs, bev_ch, bev_h, bev_w = feature.shape
                bev_feature = feature[0].reshape(bev_ch, bev_h, bev_w).detach().cpu().numpy()
                for bch in range(0, 5) :
                    channel_feature = bev_feature[bch]
                    plt.imshow(channel_feature, cmap='viridis')
                    plt.colorbar()
                    plt.grid(b=None)
                    plt.savefig(os.path.join('runs', f'{sensor}_channel_{bch}.png'))
                    plt.close()
                    
        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)
            
        else:
            assert len(features) == 1, features
            x = features[0]
        batch_size = x.shape[0]

        visualize_bev = False
        if visualize_bev :
            bs, bev_ch, bev_h, bev_w = x.shape
            bev_feature = x[0].reshape(bev_ch, bev_h, bev_w).cpu().detach().numpy()
            for bch in range(0, 5) :
                channel_feature = bev_feature[bch]
                plt.imshow(channel_feature, cmap='viridis')
                plt.colorbar()
                plt.grid(b=None)
                plt.savefig(os.path.join('runs', f'fused_channel_{bch}.png'))
                plt.close()


        x = self.decoder["backbone"](x)
        if self.decoder_neck_flag is not None : 
            x = self.decoder["neck"](x)

        visualize_bev = False
        if visualize_bev :
            # bs, bev_ch, bev_h, bev_w = x.shape
            bs, bev_ch, bev_h, bev_w = x[0].shape
            # bev_feature = x.reshape(bev_ch, bev_h, bev_w).cpu().numpy()
            bev_feature = x[0][0].reshape(bev_ch, bev_h, bev_w).cpu().detach().numpy()
            for bch in range(0, 5) :
                channel_feature = bev_feature[bch]
                plt.imshow(channel_feature, cmap='viridis')
                plt.colorbar()
                plt.grid(b=None)
                plt.savefig(os.path.join('runs', f'fused_backbone_channel_{bch}.png'))
                plt.close()


        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")

            return outputs
