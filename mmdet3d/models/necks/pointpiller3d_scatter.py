import torch
import torch.nn as nn
from mmdet.models import NECKS

@NECKS.register_module()
class PointPillarScatter3d(nn.Module):
    def __init__(self, 
                 input_shape,
                 num_bev_features):
        super().__init__()
        
        self.nx, self.ny, self.nz = input_shape
        self.num_bev_features = num_bev_features
        self.num_bev_features_before_compression = num_bev_features // self.nz

    def forward(self, inputs):
        pillar_features = inputs[0]
        coords = inputs[1]
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        #batch_size = pillar_features.shape[0]
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        return batch_spatial_features