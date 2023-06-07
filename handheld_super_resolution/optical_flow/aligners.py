# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:38:03 2023

@author: jamyl
"""

import numpy as np
from numba import cuda
from cv2 import phaseCorrelate, calcOpticalFlowFarneback
from .flow_utils import patchify_constant_flow, patchify_dense_flow
# import icaflow

# ---- Phase correlation
def phase_correlation_alignment(cuda_ref_im_grey, cuda_comp_img_grey,
                                tile_size):
    ref = cuda_ref_im_grey.copy_to_host()
    comp = cuda_comp_img_grey.copy_to_host()
    
    # flow = np.array(list(phaseCorrelate(ref, comp)[0])) # second component is the correlation : useless
    flow = np.array(list(phaseCorrelate(ref, comp)[0])) # second component is the correlation : useless
    patchwise = patchify_constant_flow(flow, tile_size, cuda_ref_im_grey.shape)
    
    return cuda.to_device(patchwise)

# ---- Farneback
def farneback_alignment(cuda_ref_im_grey, cuda_comp_img_grey,
                                tile_size):
    ref = cuda_ref_im_grey.copy_to_host()
    comp = cuda_comp_img_grey.copy_to_host()

    dense = calcOpticalFlowFarneback(ref, comp, None, 0.5, 3, 16, 3, 5, 1.2, 0)        
    
    patchwise = patchify_dense_flow(dense, tile_size)
    
    return cuda.to_device(patchwise)


# ---- ICA affinity 
def dense_ica_affinity_alignment(cuda_ref_im_grey, cuda_comp_img_grey, tile_size):
    import icaflow
    ref = cuda_ref_im_grey.copy_to_host()
    comp = cuda_comp_img_grey.copy_to_host()
    h, w = ref.shape
    
    p = icaflow.estimate_transform(ref, comp, nparams=6)
    A = np.array(((1+p[2],p[3],p[0]),(p[4],1+p[5],p[1]),(0,0,1)))
    
    dx, dy = affinity_to_flow(A, h, w)
    dense = np.stack((dx, dy), axis=-1)
    
    patchwise = patchify_dense_flow(dense, tile_size)
        
    return cuda.to_device(patchwise)

# ---- ICA global translation
def global_ica_alignment(cuda_ref_im_grey, cuda_comp_img_grey, tile_size):
    import icaflow
    ref = cuda_ref_im_grey.copy_to_host()
    comp = cuda_comp_img_grey.copy_to_host()
    h, w = ref.shape
    
    p = icaflow.estimate_transform(ref, comp, nparams=2)
    # p is directly the translation 
    flow = p.copy()
    # i have checked : this is correct
    
    patchwise = patchify_constant_flow(flow, tile_size, cuda_ref_im_grey.shape)
    
    return cuda.to_device(patchwise)









def compute_ica(u, uref, as_optical_flow=True, **kwargs):
    import icaflow
    h, w = u.shape

    p = icaflow.estimate_transform(u, uref, nparams=6, **kwargs)

    # convert 'p' from ICA conventions to a 3x3 matrix
    A = np.array(((1+p[2],p[3],p[0]),(p[4],1+p[5],p[1]),(0,0,1)))

    if as_optical_flow:
        dx, dy = affinity_to_flow(A, h, w)
        return np.stack((dx, dy), axis=-1)

    return A

def affinity_to_flow(A, h, w):
    xx, yy = identity_flow(h, w)
    return A[0,2] + xx * A[0,0] + yy * A[0,1], \
           A[1,2] + xx * A[1,0] + yy * A[1,1]
           
           
def identity_flow(h, w):
    xx, yy = np.meshgrid(np.arange(0, w), np.arange(0, h))
    return xx, yy
