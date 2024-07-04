"""Helper functions for E2E future prediction."""

import torch
import numpy as np
import copy


def bev_grids_to_coordinates(ref_grids, pc_range):
    ref_coords = copy.deepcopy(ref_grids)
    ref_coords[..., 0:1] = (ref_coords[..., 0:1] *
                            (pc_range[3] - pc_range[0]) + pc_range[0])
    ref_coords[..., 1:2] = (ref_coords[..., 1:2] *
                            (pc_range[4] - pc_range[1]) + pc_range[1])
    return ref_coords

def bev_coords_to_grids(ref_coords, bev_h, bev_w, pc_range):
    ref_grids = copy.deepcopy(ref_coords)

    ref_grids[..., 0] = ((ref_grids[..., 0] - pc_range[0]) /
                         (pc_range[3] - pc_range[0]))
    ref_grids[..., 1] = ((ref_grids[..., 1] - pc_range[1]) /
                         (pc_range[4] - pc_range[1]))
    ref_grids = ref_grids * 2 - 1.  # [-1, 1]

    # Ignore the border part.
    border_x_min = 0.5 / bev_w * 2 - 1
    border_x_max = (bev_w - 0.5) / bev_w * 2 - 1
    border_y_min = 0.5 / bev_h * 2 - 1
    border_y_max = (bev_h - 0.5) / bev_h * 2 - 1
    valid_mask = ((ref_grids[..., 0:1] > border_x_min) &
                  (ref_grids[..., 0:1] < border_x_max) &
                  (ref_grids[..., 1:2] > border_y_min) &
                  (ref_grids[..., 1:2] < border_y_max))
    return ref_grids, valid_mask

def coords_to_voxel_grids(ref_coords, bev_h, bev_w, pillar_num, pc_range):
    ref_grids = copy.deepcopy(ref_coords)

    ref_grids[..., 0] = ((ref_grids[..., 0] - pc_range[0]) /
                         (pc_range[3] - pc_range[0])) * bev_w
    ref_grids[..., 1] = ((ref_grids[..., 1] - pc_range[1]) /
                         (pc_range[4] - pc_range[1])) * bev_h
    ref_grids[..., 2] = ((ref_grids[..., 2] - pc_range[2]) /
                         (pc_range[5] - pc_range[2])) * pillar_num
    return ref_grids


def get_bev_grids(H, W, bs=1, device='cuda', dtype=torch.float, offset=0.5):
    """Get the reference points used in SCA and TSA.
    Args:
        H, W: spatial shape of bev.
        device (obj:`device`): The device where
            reference_points should be.
    Returns:
        Tensor: reference points used in decoder, has \
            shape (bs, H * W, 2).
    """
    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            offset, H - (1 - offset), H, dtype=dtype, device=device),
        torch.linspace(
            offset, W - (1 - offset), W, dtype=dtype, device=device)
    )
    ref_y = ref_y.reshape(-1)[None] / H
    ref_x = ref_x.reshape(-1)[None] / W
    ref_bev = torch.stack((ref_x, ref_y), -1)
    ref_bev = ref_bev.repeat(bs, 1, 1)
    return ref_bev


def get_bev_grids_3d(H, W, Z, bs=1, device='cuda', dtype=torch.float):
    # reference points in 3D space, used in spatial cross-attention (SCA)
    zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype,
                        device=device).view(-1, 1, 1).expand(Z, H, W) / Z
    xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                        device=device).view(1, 1, W).expand(Z, H, W) / W
    ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                        device=device).view(1, H, 1).expand(Z, H, W) / H
    ref_3d = torch.stack((xs, ys, zs), -1)
    # Z B H W ==> Z B HW ==> Z HW 3
    ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
    ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
    return ref_3d


# JIT
from torch.utils.cpp_extension import load
dvxlr = load("dvxlr", sources=[
    "third_lib/dvxlr/dvxlr.cpp",
    "third_lib/dvxlr/dvxlr.cu"], verbose=True)
class DifferentiableVoxelRenderingLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sigma, origin, points, tindex):
        pred_dist, gt_dist, dd_dsigma, indices = dvxlr.render(sigma,
                                                              origin,
                                                              points,
                                                              tindex)
        ctx.save_for_backward(dd_dsigma, indices, tindex, sigma)
        return pred_dist, gt_dist

    @staticmethod
    def backward(ctx, gradpred, gradgt):
        dd_dsigma, indices, tindex, sigma_shape = ctx.saved_tensors
        elementwise_mult = gradpred[..., None] * dd_dsigma

        invalid_grad = torch.isnan(elementwise_mult)
        elementwise_mult[invalid_grad] = 0.0

        grad_sigma = dvxlr.get_grad_sigma(elementwise_mult, indices, tindex, sigma_shape)[0]

        return grad_sigma, None, None, None


DifferentiableVoxelRendering = DifferentiableVoxelRenderingLayer.apply


# differentiable volume rendering v2.
dvxlr_v2 = load("dvxlr_v2", sources=[
    "third_lib/dvxlr/dvxlr_v2.cpp",
    "third_lib/dvxlr/dvxlr_v2.cu"], verbose=True)
class DifferentiableVoxelRenderingLayerV2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sigma, origin, points, tindex, sigma_regul):
        (pred_dist, gt_dist, dd_dsigma, indices,
         ray_pred, indicator) = dvxlr_v2.render_v2(
            sigma, origin, points, tindex, sigma_regul)
        ctx.save_for_backward(dd_dsigma, indices, tindex, sigma, indicator)
        return pred_dist, gt_dist, ray_pred, indicator

    @staticmethod
    def backward(ctx, gradpred, gradgt, grad_ray_pred, grad_indicator):
        dd_dsigma, indices, tindex, sigma_shape, indicator = ctx.saved_tensors
        elementwise_mult = gradpred[..., None] * dd_dsigma

        grad_sigma, grad_sigma_regul = dvxlr_v2.get_grad_sigma_v2(
            elementwise_mult, indices, tindex, sigma_shape, indicator, grad_ray_pred)

        return grad_sigma, None, None, None, grad_sigma_regul


DifferentiableVoxelRenderingV2 = DifferentiableVoxelRenderingLayerV2.apply


def get_inside_mask(points, point_cloud_range):
    """Get mask of points who are within the point cloud range.

    Args:
        points: A tensor with shape of [num_points, 3]
        pc_range: A list with content as
            [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    mask = ((point_cloud_range[0] <= points[..., 0]) &
            (points[..., 0] <= point_cloud_range[3]) &
            (point_cloud_range[1] <= points[..., 1]) &
            (points[..., 1] <= point_cloud_range[4]) &
            (point_cloud_range[2] <= points[..., 2]) &
            (points[..., 2] <= point_cloud_range[5]))
    return mask


from chamferdist import ChamferDistance
chamfer_distance = ChamferDistance()
def compute_chamfer_distance(pred_pcd, gt_pcd):
    loss_src, loss_dst, _ = chamfer_distance(
        pred_pcd[None, ...], gt_pcd[None, ...], bidirectional=True, reduction='sum')

    chamfer_dist_value = (loss_src / pred_pcd.shape[0]) + (loss_dst / gt_pcd.shape[0])
    return chamfer_dist_value / 2.0


def compute_chamfer_distance_inner(pred_pcd, gt_pcd, pc_range):
    pred_mask = get_inside_mask(pred_pcd, pc_range)
    inner_pred_pcd = pred_pcd[pred_mask]

    gt_mask = get_inside_mask(gt_pcd, pc_range)
    inner_gt_pcd = gt_pcd[gt_mask]

    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return 0.0

    return compute_chamfer_distance(inner_pred_pcd, inner_gt_pcd)


# visualization function for predicted point clouds.
# directly modified from nuscenes toolkit.
def _dbg_draw_pc_function(points, labels, color_map, output_path,
                          ctr=None, ctr_labels=None,):
    """Draw point cloud segmentation mask from BEV

    Args:
        points: A ndarray with shape as [-1, 3]
        labels: the label of each point with shape [-1]
        color_map: color of each label.
    """
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(1, 1, figsize=(9, 9))
    axes_limit = 40
    # points: LiDAR points with shape [-1, 3]
    viz_points = points
    dists = np.sqrt(np.sum(viz_points[:, :2] ** 2, axis=1))
    colors = np.minimum(1, dists / axes_limit / np.sqrt(2))

    # prepare color_map
    points_color = color_map[labels] / 255.  # -1, 3

    point_scale = 0.2
    scatter = ax.scatter(viz_points[:, 0], viz_points[:, 1],
                         c=points_color, s=point_scale)

    if ctr is not None:
        # draw center of the point cloud (Ego position).
        ctr_scale = 100
        ctr_color = color_map[ctr_labels] / 255.
        ax.scatter(ctr[:, 0], ctr[:, 1], c=ctr_color, s=ctr_scale, marker='x')

    ax.plot(0, 0, 'x', color='red')
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    ax.axis('off')
    ax.set_aspect('equal')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=200)


def _get_direction_of_each_query_points(points, origin=0.5):
    """
    Args:
        points: A tensor with shape as [..., 2/3] with a range of [0, 1]
        origin: The origin point position of start points.
    """
    r = points - origin
    r_norm = r / torch.sqrt((r ** 2).sum(-1, keepdims=True))
    return r_norm


import torch
import basic_utils

import numpy as np

def eye_4x4(B, device='cuda'):
    rt = torch.eye(4, device=torch.device(device)).view(1,4,4).repeat([B, 1, 1])
    return rt

def safe_inverse(a): #parallel version
    B, _, _ = list(a.shape)
    inv = a.clone()
    r_transpose = a[:, :3, :3].transpose(1,2) #inverse of rotation matrix

    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])

    return inv

def safe_inverse_single(a):
    r, t = split_rt_single(a)
    t = t.view(3,1)
    r_transpose = r.t()
    inv = torch.cat([r_transpose, -torch.matmul(r_transpose, t)], 1)
    bottom_row = a[3:4, :] # this is [0, 0, 0, 1]
    # bottom_row = torch.tensor([0.,0.,0.,1.]).view(1,4)
    inv = torch.cat([inv, bottom_row], 0)
    return inv

def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,:,0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    # xyz2 = xyz2 / xyz2[:,:,3:4]
    xyz2 = xyz2[:,:,:3]
    return xyz2

def get_camM_T_camXs(origin_T_camXs, ind=0):
    B, S = list(origin_T_camXs.shape)[0:2]
    camM_T_camXs = torch.zeros_like(origin_T_camXs)
    for b in list(range(B)):
        camM_T_origin = safe_inverse_single(origin_T_camXs[b,ind])
        for s in list(range(S)):
            camM_T_camXs[b,s] = torch.matmul(camM_T_origin, origin_T_camXs[b,s])
    return camM_T_camXs

def split_rt_single(rt):
    r = rt[:3, :3]
    t = rt[:3, 3].view(3)
    return r, t

def split_rt(rt):
    r = rt[:, :3, :3]
    t = rt[:, :3, 3].view(-1, 3)
    return r, t

def merge_rt(r, t):
    # r is B x 3 x 3
    # t is B x 3
    B, C, D = list(r.shape)
    B2, D2 = list(t.shape)
    assert(C==3)
    assert(D==3)
    assert(B==B2)
    assert(D2==3)
    t = t.view(B, 3)
    rt = eye_4x4(B, device=t.device)
    rt[:,:3,:3] = r
    rt[:,:3,3] = t
    return rt

def scale_intrinsics(K, sx, sy):
    fx, fy, x0, y0 = split_intrinsics(K)
    fx = fx*sx
    fy = fy*sy
    x0 = x0*sx
    y0 = y0*sy
    K = merge_intrinsics(fx, fy, x0, y0)
    return K

def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0

def merge_intrinsics(fx, fy, x0, y0):
    B = list(fx.shape)[0]
    K = torch.zeros(B, 4, 4, dtype=torch.float32, device=fx.device)
    K[:,0,0] = fx
    K[:,1,1] = fy
    K[:,0,2] = x0
    K[:,1,2] = y0
    K[:,2,2] = 1.0
    K[:,3,3] = 1.0
    return K

def merge_rtlist(rlist, tlist):
    B, N, D, E = list(rlist.shape)
    assert(D==3)
    assert(E==3)
    B, N, F = list(tlist.shape)
    assert(F==3)

    __p = lambda x: basic_utils.pack_seqdim(x, B)
    __u = lambda x: basic_utils.unpack_seqdim(x, B)
    rlist_, tlist_ = __p(rlist), __p(tlist)
    rtlist_ = merge_rt(rlist_, tlist_)
    rtlist = __u(rtlist_)
    return rtlist

def split_lrtlist(lrtlist):
    # splits a B x N x 19 tensor
    # into B x N x 3 (lens)
    # and B x N x 4 x 4 (rts)
    B, N, D = list(lrtlist.shape)
    assert(D==19)
    lenlist = lrtlist[:,:,:3].reshape(B, N, 3)
    ref_T_objs_list = lrtlist[:,:,3:].reshape(B, N, 4, 4)
    return lenlist, ref_T_objs_list

def merge_lrtlist(lenlist, rtlist):
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4
    # merges these into a B x N x 19 tensor
    B, N, D = list(lenlist.shape)
    assert(D==3)
    B2, N2, E, F = list(rtlist.shape)
    assert(B==B2)
    assert(N==N2)
    assert(E==4 and F==4)
    rtlist = rtlist.reshape(B, N, 16)
    lrtlist = torch.cat([lenlist, rtlist], axis=2)
    return lrtlist

def apply_4x4_to_lrtlist(Y_T_X, lrtlist_X):
    B, N, D = list(lrtlist_X.shape)
    assert(D==19)
    B2, E, F = list(Y_T_X.shape)
    assert(B2==B)
    assert(E==4 and F==4)

    lenlist, rtlist_X = split_lrtlist(lrtlist_X)

    Y_T_Xs = Y_T_X.unsqueeze(1).repeat(1, N, 1, 1)
    Y_T_Xs_ = Y_T_Xs.view(B*N, 4, 4)
    rtlist_X_ = rtlist_X.reshape(B*N, 4, 4)
    rtlist_Y_ = basic_utils.matmul2(Y_T_Xs_, rtlist_X_)
    rtlist_Y = rtlist_Y_.reshape(B, N, 4, 4)
    lrtlist_Y = merge_lrtlist(lenlist, rtlist_Y)
    return lrtlist_Y

def apply_4x4_to_lrt(Y_T_X, lrt_X):
    B, D = list(lrt_X.shape)
    assert(D==19)
    B2, E, F = list(Y_T_X.shape)
    assert(B2==B)
    assert(E==4 and F==4)

    return apply_4x4_to_lrtlist(Y_T_X, lrt_X.unsqueeze(1)).squeeze(1)

def get_xyzlist_from_lenlist(lenlist):
    B, N, D = list(lenlist.shape)
    assert(D==3)
    lx, ly, lz = torch.unbind(lenlist, axis=2)

    xs = torch.stack([lx/2., lx/2., -lx/2., -lx/2., lx/2., lx/2., -lx/2., -lx/2.], axis=2)
    ys = torch.stack([ly/2., ly/2., ly/2., ly/2., -ly/2., -ly/2., -ly/2., -ly/2.], axis=2)
    zs = torch.stack([lz/2., -lz/2., -lz/2., lz/2., lz/2., -lz/2., -lz/2., lz/2.], axis=2)

    # these are B x N x 8
    xyzlist = torch.stack([xs, ys, zs], axis=3)
    # this is B x N x 8 x 3
    return xyzlist

def get_xyzlist_from_lrtlist(lrtlist, include_clist=False):
    B, N, D = list(lrtlist.shape)
    assert(D==19)

    lenlist, rtlist = split_lrtlist(lrtlist)
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4

    xyzlist_obj = get_xyzlist_from_lenlist(lenlist)
    # xyzlist_obj is B x N x 8 x 3

    rtlist_ = rtlist.reshape(B*N, 4, 4)
    xyzlist_obj_ = xyzlist_obj.reshape(B*N, 8, 3)
    xyzlist_cam_ = apply_4x4(rtlist_, xyzlist_obj_)
    xyzlist_cam = xyzlist_cam_.reshape(B, N, 8, 3)

    if include_clist:
        clist_cam = get_clist_from_lrtlist(lrtlist).unsqueeze(2)
        xyzlist_cam = torch.cat([xyzlist_cam, clist_cam], dim=2)
    return xyzlist_cam

def get_clist_from_lrtlist(lrtlist):
    B, N, D = list(lrtlist.shape)
    assert(D==19)

    lenlist, rtlist = split_lrtlist(lrtlist)
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4

    xyzlist_obj = torch.zeros((B, N, 1, 3), device=lrtlist.device)
    # xyzlist_obj is B x N x 8 x 3

    rtlist_ = rtlist.reshape(B*N, 4, 4)
    xyzlist_obj_ = xyzlist_obj.reshape(B*N, 1, 3)
    xyzlist_cam_ = apply_4x4(rtlist_, xyzlist_obj_)
    xyzlist_cam = xyzlist_cam_.reshape(B, N, 3)
    return xyzlist_cam

def wrap2pi(rad_angle):
    # puts the angle into the range [-pi, pi]
    return torch.atan2(torch.sin(rad_angle), torch.cos(rad_angle))

def xyd2pointcloud(xyd, pix_T_cam):
    # xyd is like a pointcloud but in pixel coordinates;
    # this means xy comes from a meshgrid with bounds H, W, 
    # and d comes from a depth map
    B, N, C = list(xyd.shape)
    assert(C==3)
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = pixels2camera(xyd[:,:,0], xyd[:,:,1], xyd[:,:,2], fx, fy, x0, y0)
    return xyz

def pixels2camera(x,y,z,fx,fy,x0,y0):
    # x and y are locations in pixel coordinates, z is a depth in meters
    # they can be images or pointclouds
    # fx, fy, x0, y0 are camera intrinsics
    # returns xyz, sized B x N x 3

    B = x.shape[0]
    
    fx = torch.reshape(fx, [B,1])
    fy = torch.reshape(fy, [B,1])
    x0 = torch.reshape(x0, [B,1])
    y0 = torch.reshape(y0, [B,1])

    x = torch.reshape(x, [B,-1])
    y = torch.reshape(y, [B,-1])
    z = torch.reshape(z, [B,-1])
    
    # unproject
    x = (z/fx)*(x-x0)
    y = (z/fy)*(y-y0)
    
    xyz = torch.stack([x,y,z], dim=2)
    # B x N x 3
    return xyz

def camera2pixels(xyz, pix_T_cam):
    # xyz is shaped B x H*W x 3
    # returns xy, shaped B x H*W x 2
    
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    x, y, z = torch.unbind(xyz, dim=-1)
    B = list(z.shape)[0]

    fx = torch.reshape(fx, [B,1])
    fy = torch.reshape(fy, [B,1])
    x0 = torch.reshape(x0, [B,1])
    y0 = torch.reshape(y0, [B,1])
    x = torch.reshape(x, [B,-1])
    y = torch.reshape(y, [B,-1])
    z = torch.reshape(z, [B,-1])

    EPS = 1e-4
    z = torch.clamp(z, min=EPS)
    x = (x*fx)/z + x0
    y = (y*fy)/z + y0
    xy = torch.stack([x, y], dim=-1)
    return xy
