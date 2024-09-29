import numpy as np
import torch
from spconv.tools import CUDAKernelTimer

def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.

    this function don't contain except handle code. so use this carefully when
    indice repeats, don't support repeat add which is supported in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1] :]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret


class SparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size, grid=None, voxel_num=None, benchmark=False, force_algo=None, enable_timer=False):
        """
        Args:
            grid: pre-allocated grid tensor.
                  should be used when the volume of spatial shape
                  is very large.
        """
        self.features = features
        self.indices = indices
        if self.indices.dtype != torch.int32:
            self.indices.int()
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        self.indice_dict = {}
        self.grid = grid
        self.voxel_num = voxel_num
        self.benchmark = benchmark
        self.benchmark_record = {}
        self.thrust_allocator = None
        self.force_algo = force_algo
        self.int8_scale = None
        self._timer = CUDAKernelTimer(enable_timer)
    @property
    def spatial_size(self):
        return np.prod(self.spatial_shape)

    def find_indice_pair(self, key):
        if key is None:
            return None
        if key in self.indice_dict:
            return self.indice_dict[key]
        return None

    def dense(self, channels_first=True):
        output_shape = (
            [self.batch_size] + list(self.spatial_shape) + [self.features.shape[1]]
        )
        res = scatter_nd(self.indices.long(), self.features, output_shape)
        if not channels_first:
            return res
        ndim = len(self.spatial_shape)
        trans_params = list(range(0, ndim + 1))
        trans_params.insert(1, ndim + 1)
        return res.permute(*trans_params).contiguous()

    @property
    def sparity(self):
        return self.indices.shape[0] / np.prod(self.spatial_shape) / self.batch_size
    
    @property
    def is_quantized(self):
        return self.features.dtype == torch.qint8

    def q_scale(self):
        if self.is_quantized:
            return self.features.q_scale()
        raise ValueError("sparse tensor must be quantized")
    
    def shadow_copy(self) -> "SparseConvTensor":
        """create a new spconv tensor with all member unchanged"""
        tensor = SparseConvTensor(self.features, self.indices,
                                  self.spatial_shape, self.batch_size,
                                  self.grid, self.voxel_num, self.benchmark)
        tensor.benchmark = self.benchmark
        tensor.indice_dict = self.indice_dict
        tensor.benchmark_record = self.benchmark_record
        tensor.thrust_allocator = self.thrust_allocator
        tensor._timer = self._timer
        tensor.force_algo = self.force_algo
        tensor.int8_scale = self.int8_scale
        return tensor
    
    def replace_feature(self, new_features):
        self.features = new_features
        return self
    
    def half(self) :
        self.features = self.features.half()
        return self