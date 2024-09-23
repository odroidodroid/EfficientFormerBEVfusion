import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from mmdet3d.models.builder import VTRANSFORMS
from .geom_utils import *
from .basic_utils import *
from .vox_utils import *
from mmcv.runner import auto_fp16, force_fp32
from .latent_rendering import LatentRendering
__all__ = ["SimpleBEV"]
@VTRANSFORMS.register_module()
class SimpleBEV(nn.Module) :
    def __init__(self, image_size, feature_bev_size, bounds,
                 use_radar=False,
                 use_lidar=False,
                 use_metaradar=False,
                 do_rgbcompress=True,
                 rand_flip=False,
                 latent_dim=128,
                 latent_render=False):
        super(SimpleBEV, self).__init__()

        # scene_centroid_x = 0.5
        # scene_centroid_y = 0.0
        # scene_centroid_z = 0.5
        scene_centroid_x = 0.0
        scene_centroid_y = 0.1
        scene_centroid_z = 0.0

        scene_centroid_py = np.array([scene_centroid_x,
                                    scene_centroid_y,
                                    scene_centroid_z]).reshape([1, 3])
        scene_centroid = torch.from_numpy(scene_centroid_py).float()

        self.image_size = image_size
        self.Z, self.Y, self.X = feature_bev_size # 200, 8, 200
        self.use_radar = use_radar
        self.use_lidar = use_lidar
        self.use_metaradar = use_metaradar
        self.do_rgbcompress = do_rgbcompress   
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.latent_render = latent_render
        if latent_render :
            self.latent_rendering = LatentRendering() # ???
        else :
            self.latent_rendering = None
        
        # # Encoder
        self.feat2d_dim = feat2d_dim = latent_dim

        # BEV compressor
        if self.use_radar:
            if self.use_metaradar:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*self.Y + 16*self.Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*self.Y+1, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
        elif self.use_lidar:
            self.bev_compressor = nn.Sequential(
                nn.Conv2d(feat2d_dim*self.Y+self.Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(latent_dim),
                nn.GELU(),
            )
        else:
            if self.do_rgbcompress:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*self.Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:
                # use simple sum
                pass

        self.vox_utils = Vox_util(self.Z, self.Y, self.X,
                                        scene_centroid=scene_centroid,
                                        bounds=bounds,
                                        assert_cube=False)
        # self.xyz_memA = gridcloud3d(1, self.Z, self.Y, self.X, norm=False)
        # self.xyz_camA = self.vox_utils.Mem2Ref(self.xyz_memA, self.Z, self.Y, self.X, assert_cube=False)
        self.xyz_camA = None
        
    @force_fp32(apply_to="cam_feat")
    def forward(self, cam_feat, points, camera2ego, lidar2ego, lidar2camera, lidar2image, cam_intrinsic, camera2lidar, img_aug_matrix, lidar_aug_matrix, img_metas,**kwargs):
        '''
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_camXs: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cam0_T_camXs: (B,S,4,4)
        vox_util: vox util object
        rad_occ_mem0:
            - None when use_radar = False, use_lidar = False
            - (B, 1, Z, Y, X) when use_radar = True, use_metaradar = False
            - (B, 16, Z, Y, X) when use_radar = True, use_metaradar = True
            - (B, 1, Z, Y, X) when use_lidar = True
        '''
        
        B, S, C, _, _ = cam_feat.shape
        H, W = self.image_size
        # reshape tensors
        __p = lambda x: pack_seqdim(x, B)
        __u = lambda x: unpack_seqdim(x, B)
        # rgb_camXs_ = __p(rgb_camXs)
        intrins = kwargs['sim_intrins'].data[0].cuda()
        rots = kwargs['sim_rots'].data[0].cuda()
        trans = kwargs['sim_trans'].data[0].cuda()
        if not self.training :
            rots = rots.unsqueeze(0)
            trans = trans.unsqueeze(0)
            intrins = intrins.unsqueeze(0)
        velo_T_cams = merge_rtlist(rots, trans)
        cam0_T_camXs = get_camM_T_camXs(velo_T_cams, ind=0)
        intrins = __p(intrins)
        pix_T_cams_ = merge_intrinsics(*split_intrinsics(intrins))
        cam0_T_camXs_ = __p(cam0_T_camXs)
        camXs_T_cam0_ = safe_inverse(cam0_T_camXs_)

        feat_camXs_ = __p(cam_feat)
        _, C, Hf, Wf = feat_camXs_.shape

        sy = Hf/float(H)
        sx = Wf/float(W)
        Z, Y, X = self.Z, self.Y, self.X

        # unproject image feature to 3d grid
        featpix_T_cams_ = scale_intrinsics(pix_T_cams_, sx, sy)
        if self.xyz_camA is not None:
            xyz_camA = self.xyz_camA.to(feat_camXs_.device).repeat(B*S,1,1)
        else:
            xyz_camA = None
        feat_mems_ = self.vox_utils.unproject_image_to_mem(
            feat_camXs_,
            matmul2(featpix_T_cams_, camXs_T_cam0_),
            camXs_T_cam0_, Z, Y, X,
            xyz_camA=xyz_camA)
        feat_mems = __u(feat_mems_) # B, S, C, Z, Y, X

        mask_mems = (torch.abs(feat_mems) > 0).float()
        feat_mem = reduce_masked_mean(feat_mems, mask_mems, dim=1) # B, C, Z, Y, X

        # bev compressing
        if self.do_rgbcompress:
            feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
            feat_bev = self.bev_compressor(feat_bev_)
        else:
            feat_bev = torch.sum(feat_mem, dim=3)
        return feat_bev