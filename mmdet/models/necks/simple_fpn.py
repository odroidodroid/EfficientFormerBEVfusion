# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import Tensor
import torch
from ..builder import NECKS


@NECKS.register_module()
class SimpleFPN(BaseModule):
    """Simple Feature Pyramid Network for ViTDet."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 no_norm_on_lateral=False,
                 conv_cfg = None,
                 norm_cfg = None,
                 act_cfg = None,
                 upsample_cfg=dict(mode="bilinear", align_corners=True),
                 ) -> None:
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsampling_layer = UpsamplingConcat(1536, 512)
        self.depth_layer = nn.Conv2d(512, out_channels, kernel_size=1, padding=0)
    @auto_fp16()
    def forward(self, inputs) :
        outs = self.upsampling_layer(inputs[1], inputs[0])
        outs = self.depth_layer(outs)
        return outs

class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)
