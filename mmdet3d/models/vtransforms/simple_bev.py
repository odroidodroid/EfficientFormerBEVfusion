import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from mmdet3d.models.builder import VTRANSFORMS
import geom_utils
import basic_utils
import vox_utils

from latent_rendering import LatentRendering
__all__ = ["SimpleBEV"]
@VTRANSFORMS.register_module()
class SimpleBEV(nn.Module) :
    def __init__(self, image_size, feature_bev_size, 
                 use_radar=False,
                 use_lidar=False,
                 use_metaradar=False,
                 do_rgbcompress=True,
                 rand_flip=False,
                 latent_dim=128,
                 latent_render=False):
        super(SimpleBEV, self).__init__()

        scene_centroid_x = 0.0
        scene_centroid_y = 1.0
        scene_centroid_z = 0.0

        scene_centroid_py = np.array([scene_centroid_x,
                                    scene_centroid_y,
                                    scene_centroid_z]).reshape([1, 3])
        scene_centroid = torch.from_numpy(scene_centroid_py).float()

        self.image_size = image_size
        self.X, self.Y, self.Z = feature_bev_size # 200, 8, 200 -> 180, 80, 180????
        self.use_radar = use_radar
        self.use_lidar = use_lidar
        self.use_metaradar = use_metaradar
        self.do_rgbcompress = do_rgbcompress   
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        if latent_render :
            self.latent_rendering = LatentRendering(embed_dims=latent_dim) # ???
        else :
            self.latent_rendering = None
        # self.encoder_type = encoder_type

        # self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float().cuda()
        # self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float().cuda()
        
        # # Encoder
        self.feat2d_dim = feat2d_dim = latent_dim
        # if encoder_type == "res101":
        #     self.encoder = Encoder_res101(feat2d_dim)
        # elif encoder_type == "res50":
        #     self.encoder = Encoder_res50(feat2d_dim)
        # elif encoder_type == "effb0":
        #     self.encoder = Encoder_eff(feat2d_dim, version='b0')
        # else:
        #     # effb4
        #     self.encoder = Encoder_eff(feat2d_dim, version='b4')

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

        # Decoder
        # self.decoder = Decoder(
        #     in_channels=latent_dim,
        #     n_classes=1,
        #     predict_future_flow=False
        # )

        # Weights
        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            
        # set_bn_momentum(self, 0.1)
        self.vox_utils = vox_utils.Vox_util(self.Z, self.Y, self.X,
                                        scene_centroid=scene_centroid,
                                        assert_cube=False)
        self.xyz_memA = basic_utils.gridcloud3d(1, self.Z, self.Y, self.X, norm=False)
        self.xyz_camA = self.vox_utils.Mem2Ref(self.xyz_memA, self.Z, self.Y, self.X, assert_cube=False)
        
    def forward(self, cam_feat, sensor2ego, lidar2ego, lidar2image, cam_intrinsic, camera2lidar):
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
        
        # pix_T_cams : ????
        # cam0_T_camXs : ????
        # rad_occ_mem0=None

        rots = sensor2ego[..., :3, :3]
        trans = sensor2ego[..., :3, 3]
        intrins = cam_intrinsic[..., :3, :3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        
        B, S, C, H, W = self.image_size
        # assert(C==3)
        # # reshape tensors
        __p = lambda x: basic_utils.pack_seqdim(x, B)
        __u = lambda x: basic_utils.unpack_seqdim(x, B)
        # rgb_camXs_ = __p(rgb_camXs)
        pix_T_cams_ = __p(pix_T_cams)
        cam0_T_camXs_ = __p(cam0_T_camXs)
        camXs_T_cam0_ = geom_utils.safe_inverse(cam0_T_camXs_)

        # # rgb encoder
        # device = rgb_camXs_.device
        # rgb_camXs_ = (rgb_camXs_ + 0.5 - self.mean.to(device)) / self.std.to(device)
        # if self.rand_flip:
        #     B0, _, _, _ = rgb_camXs_.shape
        #     self.rgb_flip_index = np.random.choice([0,1], B0).astype(bool)
        #     rgb_camXs_[self.rgb_flip_index] = torch.flip(rgb_camXs_[self.rgb_flip_index], [-1])
        # feat_camXs_ = self.encoder(rgb_camXs_)
        # if self.rand_flip:
        #     feat_camXs_[self.rgb_flip_index] = torch.flip(feat_camXs_[self.rgb_flip_index], [-1])
        _, C, Hf, Wf = cam_feat.shape

        sy = Hf/float(H)
        sx = Wf/float(W)
        Z, Y, X = self.Z, self.Y, self.X

        # unproject image feature to 3d grid
        featpix_T_cams_ = geom_utils.scale_intrinsics(pix_T_cams_, sx, sy)
        if self.xyz_camA is not None:
            xyz_camA = self.xyz_camA.to(cam_feat.device).repeat(B*S,1,1)
        else:
            xyz_camA = None
        feat_mems_ = self.vox_utils.unproject_image_to_mem(
            cam_feat,
            basic_utils.matmul2(featpix_T_cams_, camXs_T_cam0_),
            camXs_T_cam0_, Z, Y, X,
            xyz_camA=xyz_camA)
        feat_mems = __u(feat_mems_) # B, S, C, Z, Y, X

        mask_mems = (torch.abs(feat_mems) > 0).float()
        feat_mem = basic_utils.reduce_masked_mean(feat_mems, mask_mems, dim=1) # B, C, Z, Y, X

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0,1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0,1], B).astype(bool)
            feat_mem[self.bev_flip1_index] = torch.flip(feat_mem[self.bev_flip1_index], [-1])
            feat_mem[self.bev_flip2_index] = torch.flip(feat_mem[self.bev_flip2_index], [-3])

            # if rad_occ_mem0 is not None:
            #     rad_occ_mem0[self.bev_flip1_index] = torch.flip(rad_occ_mem0[self.bev_flip1_index], [-1])
            #     rad_occ_mem0[self.bev_flip2_index] = torch.flip(rad_occ_mem0[self.bev_flip2_index], [-3])

        # bev compressing
        # if self.use_radar:
        #     assert(rad_occ_mem0 is not None)
        #     if not self.use_metaradar:
        #         feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
        #         rad_bev = torch.sum(rad_occ_mem0, 3).clamp(0,1) # squish the vertical dim
        #         feat_bev_ = torch.cat([feat_bev_, rad_bev], dim=1)
        #         feat_bev = self.bev_compressor(feat_bev_)
        #     else:
        #         feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
        #         rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 16*Y, Z, X)
        #         feat_bev_ = torch.cat([feat_bev_, rad_bev_], dim=1)
        #         feat_bev = self.bev_compressor(feat_bev_)
        # elif self.use_lidar:
        #     assert(rad_occ_mem0 is not None)
        #     feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
        #     rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, Y, Z, X)
        #     feat_bev_ = torch.cat([feat_bev_, rad_bev_], dim=1)
        #     feat_bev = self.bev_compressor(feat_bev_)
        # else: # rgb only
        if self.do_rgbcompress:
            feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
            feat_bev = self.bev_compressor(feat_bev_)
        else:
            feat_bev = torch.sum(feat_mem, dim=3)

        if not isinstance(self.latent_rendering, None) :
            feat_bev = self.latent_rendering(feat_bev.permute(0, 2, 3, 1)) # bs, bev_h, bev_w, embed_dim


        # bev decoder
        # out_dict = self.decoder(feat_bev, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)

        # raw_e = out_dict['raw_feat']
        # feat_e = out_dict['feat']
        # seg_e = out_dict['segmentation']
        # center_e = out_dict['instance_center']
        # offset_e = out_dict['instance_offset']

        # return raw_e, feat_e, seg_e, center_e, offset_e
        return feat_bev