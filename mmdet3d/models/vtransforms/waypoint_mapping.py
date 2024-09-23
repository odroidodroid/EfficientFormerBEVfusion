from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.ops import bev_pool
from mmdet3d.models.builder import VTRANSFORMS
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["WayPointMapping"]


@VTRANSFORMS.register_module()
class WayPointMapping(nn.Module) :
    def __init__(
        self,
        in_channels,
        out_channels,
        image_size,
        feature_size,
        bev_feature_size=(180, 180),
        rbound=[10,20,30,40,50],
        num_points=100
    )-> None :
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.bev_feature_size = bev_feature_size
        self.rbound = rbound
        #self.waypoints = self.create_waypoints(rbound, num_points)

    def create_waypoints(self, rbound, num_points) :
        pass

    def get_geometry(self, waypoints, lidar2image) :
        pass
    
    def scale_intrinsics(self, lidar2image, scale) :
        scaled_matrix = lidar2image.clone()
        scaled_matrix[0, 0] /= scale
        scaled_matrix[1, 1] /= scale
        scaled_matrix[0, 2] /= scale
        scaled_matrix[1, 2] /= scale
        return scaled_matrix

    def forward(self, 
                img_feat, 
                points,
                sensor2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                cam_intrinsic,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                **kwargs) :
        # point projection
        # rots = sensor2ego[..., :3, :3]
        # trans = sensor2ego[..., :3, 3]
        # intrins = cam_intrinsic[..., :3, :3]
        # post_rots = img_aug_matrix[..., :3, :3]
        # post_trans = img_aug_matrix[..., :3, 3]
        # lidar2ego_rots = lidar2ego[..., :3, :3]
        # lidar2ego_trans = lidar2ego[..., :3, 3]
        # camera2lidar_rots = camera2lidar[..., :3, :3]
        # camera2lidar_trans = camera2lidar[..., :3, 3]

        batch_size = len(points)
        # depth = torch.zeros(batch_size, img_feat.shape[1], 1, *self.image_size).to(
        #     points[0].device
        # )
        bev_feature_list = []
        for b in range(batch_size):

            bev_feature_c = torch.zeros((img_feat.shape[2], self.bev_feature_size[0], self.bev_feature_size[1]), device=img_feat.device).half()


            cur_coords = points[b][:, :3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )
            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            # for c in range(on_img.shape[0]):
            #     masked_coords = cur_coords[c, on_img[c]].long()
            #     masked_dist = dist[c, on_img[c]]
            #     depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist
            # 
            
            # make u,v,d coordinate
            uvd_coords = torch.cat((cur_coords, dist.unsqueeze(-1)), dim=-1)
            u,v = uvd_coords[..., 0]/8.0, uvd_coords[..., 1]/8.0
            d = uvd_coords[..., 2]
            
            # get feature from feature map - 1/8 downsampled x,y
            for c in range(on_img.shape[0]) :            
                u_int = u[c, on_img[c]]
                v_int = v[c, on_img[c]]
                d_int = d[c, on_img[c]]
                # uvd -> xyd
                cur_lidar2image = self.scale_intrinsics(cur_lidar2image, 8.0)
                z = d_int
                x = (u_int * z).unsqueeze(-1) #### TODO uvd[..., 0] ??
                y = (v_int * z).unsqueeze(-1) #### TODO uvd[..., 1] ??
                z = z.unsqueeze(-1)
                xyz_coords = torch.cat([x,y,z], dim=-1)
                xyz_coords -= cur_lidar2image[c, :3, 3].reshape(-1, 3)
                xyz_coords = torch.inverse(cur_lidar2image[c, :3, :3]).matmul(xyz_coords.transpose(-1, 0))
                xyz_coords = xyz_coords.transpose(-1, 0)
                z = xyz_coords[..., 2]
                x = xyz_coords[..., 0].long()
                y = xyz_coords[..., 1].long()

                valid_mask_xyz = (z >= 0) & (x >= -54) & (y >= -54) & (x < 54) & (y < 54)
                valid_mask_uv = (u_int >= 0) & (u_int < self.image_size[0]//8) & (v_int >= 0) & (v_int < self.image_size[1]//8)
                valid_mask = valid_mask_uv & valid_mask_xyz
                
                u_int = u_int[valid_mask].long()
                v_int = v_int[valid_mask].long()
                x_int = (x[valid_mask] / 0.3).long()
                y_int = (y[valid_mask] / 0.3).long()

                # xyd -> xyz
                img_feat_xyz = torch.zeros((img_feat.shape[2], self.bev_feature_size[0], self.bev_feature_size[1]), device=img_feat.device).half() # TODO : shape? bev feature size or ...
                img_feat_xyz[:, x_int, y_int] = img_feat[b, c, :, u_int, v_int] # TODO : xyz? yxz?
                bev_feature_c += img_feat_xyz
            # xyz points to bev
            bev_feature_list.append(bev_feature_c)
            # TODO : how to make bev feature size? or ready to cross attention?
        bev_feature = torch.stack(bev_feature_list)
        return bev_feature