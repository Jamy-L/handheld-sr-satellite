# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:35:54 2023

@author: jamyl
"""

import numpy as np
from numba import cuda, float32
import math

from handheld_super_resolution.utils import DEFAULT_THREADS
from handheld_super_resolution.linalg import bilinear_interpolation

TAG_CHAR = np.array([202021.25], np.float32)


def densify_patchwise_flow(flow, tile_size, imshape):
    imsize_y, imsize_x = imshape[:2]

    dense = cuda.device_array(imshape + (2,), np.float64)

    blocks_x = math.ceil(imsize_x / DEFAULT_THREADS)
    blocks_y = math.ceil(imsize_y / DEFAULT_THREADS)

    blocks = (blocks_x, blocks_y, 2)  # 2 components
    cuda_densify_patchwise_flow[blocks, (DEFAULT_THREADS, DEFAULT_THREADS, 1)](
        cuda.to_device(flow), tile_size, dense)
    return dense.copy_to_host()


@cuda.jit()
def cuda_densify_patchwise_flow(flow, tile_size, dense):
    x, y, c = cuda.grid(3)
    imsize_y, imsize_x, _ = dense.shape

    if not (0 <= x < imsize_x
            and 0 <= y < imsize_y):
        return

    tile_y = y // tile_size
    tile_x = x // tile_size

    dense[y, x, c] = flow[tile_y, tile_x, c]


def patchify_dense_flow(dense, tile_size):
    imsize_y, imsize_x, _ = dense.shape

    n_patch_y = math.ceil(imsize_y / tile_size)
    n_patch_x = math.ceil(imsize_x / tile_size)

    flow = cuda.device_array((n_patch_y, n_patch_x, 2), np.float64)

    blocks_x = math.ceil(n_patch_x / DEFAULT_THREADS)
    blocks_y = math.ceil(n_patch_y / DEFAULT_THREADS)

    blocks = (blocks_x, blocks_y, 2)  # 2 components

    cuda_patchify_dense_flow[blocks, (DEFAULT_THREADS, DEFAULT_THREADS, 1)](
        cuda.to_device(dense), tile_size, flow)

    return flow.copy_to_host()


@cuda.jit()
def cuda_patchify_dense_flow(dense, tile_size, flow):
    patch_x, patch_y, c = cuda.grid(3)

    n_patch_y, n_patch_x, _ = flow.shape

    if not (0 <= patch_x < n_patch_x
            and 0 <= patch_y < n_patch_y):
        return

    u = 0
    v = 0
    c = 0
    for i in range(tile_size):
        for j in range(tile_size):
            y = patch_y * tile_size + j
            x = patch_x * tile_size + i

            if (0 <= x < dense.shape[1]
                    and 0 <= y < dense.shape[0]):
                u += dense[y, x, 0]
                v += dense[y, x, 1]
                c += 1

    flow[patch_y, patch_x, 0] = u / c
    flow[patch_y, patch_x, 1] = v / c


def densify_constant_flow(flow, imshape):
    assert flow.size == 2, "flow is not constant"

    patchwise = np.zeros(imshape + (2,), np.float32)
    patchwise += flow[None, None]
    
    return patchwise

def patchify_constant_flow(constant_flow, tile_size, imshape):
    dense = densify_constant_flow(constant_flow, imshape)
    patchwise = patchify_dense_flow(dense, tile_size)
    
    return patchwise    


def writeFlow(filename, uv, v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.

    https://github.com/NVIDIA/flownet2-pytorch/blob/master/utils/flow_utils.py

    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()
    
def warp_image(image, dense_flow):
    assert len(image.shape)==2
    assert image.shape == dense_flow.shape[:2]
    
    warped = cuda.device_array(image.shape, np.float32)
    
    threads = (DEFAULT_THREADS, DEFAULT_THREADS)
    blocksx = math.ceil(image.shape[1]/threads[1])
    blocksy = math.ceil(image.shape[0]/threads[0])
    blocks = (blocksx, blocksy)
    
    cuda_warp_image[blocks, threads](image, dense_flow, warped)
    
    return warped.copy_to_host()


@cuda.jit()
def cuda_warp_image(image, dense_flow, warped):
    x, y = cuda.grid(2)
    if not (0 <= x < image.shape[1] and
            0 <= y < image.shape[0]):
        return
    
    x_new = x + dense_flow[y, x, 0]
    y_new = y + dense_flow[y, x, 1]
    
    if not (0 <= x_new < image.shape[1]-1 and
            0 <= y_new < image.shape[0]-1):
        warped[y, x] = 0
        return
    
    normalised_pos_x, floor_x = math.modf(x_new) # https://www.rollpie.com/post/252
    normalised_pos_y, floor_y = math.modf(y_new) # separating floor and floating part
    floor_x = int(floor_x)
    floor_y = int(floor_y)
    
    ceil_x = floor_x + 1
    ceil_y = floor_y + 1
    
    pos = cuda.local.array(2, float32)
    pos[0] = normalised_pos_y
    pos[1] = normalised_pos_x
    
    buffer_val = cuda.local.array((2, 2), float32)
    buffer_val[0, 0] = image[floor_y, floor_x]
    buffer_val[0, 1] = image[floor_y, ceil_x]
    buffer_val[1, 0] = image[ceil_y, floor_x]
    buffer_val[1, 1] = image[ceil_y, ceil_x]
    
    interpolated_val = bilinear_interpolation(buffer_val, pos)
    
    warped[y, x] = interpolated_val




