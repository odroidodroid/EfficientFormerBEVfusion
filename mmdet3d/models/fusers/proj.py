from typing import List

import torch
from torch import nn
import numpy as np
from mmdet3d.models.builder import FUSERS
#from ops.deform_attn import deform_attn
from pyquaternion import Quaternion

__all__ = ["ProjFuser"]


@FUSERS.register_module()
class ProjFuser(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(self, points, img_feat, lid2img=None, ego2lid=None, img2lid=None, img_h=None, img_w=None) -> torch.Tensor:
        B, BW, BH, T, C, _ = points.shape ## BEV features or real lidar points ? real lidar points
        N = 6

        self.lid2img = torch.from_numpy(np.asarray(lid2img).astype(np.float32)).to(device=img_feat.device)
        self.ego2lid = torch.from_numpy(np.asarray(ego2lid).astype(np.float32)).to(device=img_feat.device)
        self.img2lid = torch.from_numpy(np.asarray(img2lid).astype(np.float32)).to(device=img_feat.device)
        self.img_h = img_h
        self.img_w = img_w
        self.ego2lid2img = torch.matmul(lid2img, ego2lid)

        
        img2lidar = self.img2lid
        # 1. transform pcd to ego vehicle, 2. transform from ego to global, 3. transform from global into ego frame of image         
        lidar2img = self.ego2lid2img[:, :(T*N), None, None, :, :] 
        lidar2img = lidar2img.expand()
        lidar2img = lidar2img.reshape()
        
        ones = torch.ones_like(points[..., :1])
        points = torch.cat([points, ones], dim=-1)
        points = points.expand()
        points = points.transpose()
        
        proj_points = torch.matmul(lidar2img, points).squeeze(-1) # lidar to image projection
        proj_points = deform_attn(proj_points, img_feat) # fuse, more layer ?
        
        fused_points = torch.matmul(img2lidar, proj_points) # lidar image projection to lidar
        return fused_points
