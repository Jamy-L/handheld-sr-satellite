# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:38:07 2022

This script contains : 
    - The implementation of Alg. 4, the conventionnal accumulation
    - The implementation of Alg. 11, where the reference image is merged


@author: jamyl
"""


import math

import numpy as np
from numba import uint8, cuda
import torch as th
from torchvision.transforms import GaussianBlur


from HDR_DSP_SR.warpingOperator import WarpedLoss
from HDR_DSP_SR.models import FNet
from .utils import clamp, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_NUMPY_FLOAT_TYPE, EPSILON_DIV, DEFAULT_THREADS
from .utils_image import denoise_power_merge, denoise_range_merge, zoombase_weighted, flowEstimation, base_detail_decomp_
from .linalg import quad_mat_prod, invert_2x2, interpolate_cov

def merge_bases(ref_img, comp_imgs,
                ref_exposure, exposures, sigma_bd=1):
    
    Fnet = FNet().float().to('cuda')
    checkpoint = th.load("HDR_DSP_SR/syntheticPretrainedModel/Model_ME.pth.tar", map_location=th.device('cpu'))
    Fnet.load_state_dict(checkpoint['state_dictFnet']) 
    Fnet.eval()
    
    with th.no_grad():
        gaussian_filter = GaussianBlur(11, sigma=sigma_bd).to('cuda')
        
        th_burst = th.from_numpy(np.concatenate((ref_img[None], comp_imgs), axis=0))[None].float().to('cuda')
        th_expotime = th.from_numpy(np.concatenate((ref_exposure[None], exposures), axis=0))[None, :, None, None].float().to('cuda')
        th_burst *= th_expotime # handheld input is predivided. multiplying to fit deepnet framework.
        
        warping = WarpedLoss(interpolation = 'bicubicTorch') 
        flow = flowEstimation(th_burst/th_expotime*3.4, ME=Fnet, gaussian_filter = gaussian_filter, warping = warping, device='cuda')
        
        
        th_bases, th_details = base_detail_decomp_(th_burst/th_expotime, gaussian_filter)
        
        # cpu_ref_detail = th_details[0, 0].cpu().numpy()
        # ref_detail = cuda.to_device(cpu_ref_detail)
        
        ref_detail = cuda.as_cuda_array(th_details[0, 0])
        
        
        # SR for ref base
        warping = WarpedLoss(interpolation = 'bicubicTorch') 
        SR_base = zoombase_weighted(th_bases, th_expotime, flow.contiguous(), 'cuda', warping=warping)
        SR_base = cuda.as_cuda_array(SR_base[0, 0])
        
        return SR_base, ref_detail
        
def merge_ref(ref_img, kernels, ref_exposure, num, den, options, params, acc_rob=None):
    """
    Implementation of Alg. 11: AccumulationReference
    Accumulates the reference frame into num and den, while considering
    (if enabled) the accumulated robustness mask to enforce single frame SR if
    necessary.

    Parameters
    ----------
    ref_img : device Array[imshape_y, imshape_x]
        Reference image J_1
    kernels : device Array[imshape_y//2, imshape_x//2, 2, 2]
        Covariance Matrices Omega_1
    num : device Array[s*imshape_y, s*imshape_x]
        Numerator of the accumulator
    den : device Array[s*imshape_y, s*imshape_x]
        Denominator of the accumulator
    options : dict
        verbose options.
    params : dict
        parameters (containing the zoom s).
    acc_rob : [imshape_y//2, imshape_x//2], optional
        accumulated robustness mask. The default is None.

    Returns
    -------
    None.

    """
    scale = params['scale']
    
    CFA_pattern = cuda.to_device(params['exif']['CFA Pattern'])
    bayer_mode = params['mode'] == 'bayer'
    iso_kernel = params['kernel'] == 'iso'
    
    # moving noise model to GPU
    cuda_std_curve = cuda.to_device(params['std_curve'])
    
    robustness_denoise = params['accumulated robustness denoiser']['on']
    # numba is strict on types and dimension : let's use a consistent object
    # for acc_rob even when it is not used.
    if robustness_denoise:
        rad_max = params['accumulated robustness denoiser']['rad max']
        max_multiplier = params['accumulated robustness denoiser']['max multiplier']
        max_frame_count = params['accumulated robustness denoiser']['max frame count']
    else:
        acc_rob = cuda.device_array((1,1), DEFAULT_NUMPY_FLOAT_TYPE)
        rad_max = 0
        max_multiplier = 0.
        max_frame_count = 0
    
    
    output_shape_y, output_shape_x, _ = num.shape

    # dispatching threads. 1 thread for 1 output pixel
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS) # maximum, we may take less
    
    blockspergrid_x = math.ceil(output_shape_x/threadsperblock[1])
    blockspergrid_y = math.ceil(output_shape_y/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    accumulate_ref[blockspergrid, threadsperblock](
        ref_img, kernels, ref_exposure, cuda_std_curve,
        bayer_mode, iso_kernel, scale, CFA_pattern,
        num, den,
        acc_rob, robustness_denoise, max_frame_count, rad_max, max_multiplier)
    
    
@cuda.jit
def accumulate_ref(ref_img, covs, ref_exposure, std_curve,
                   bayer_mode, iso_kernel, scale, CFA_pattern,
                   num, den, acc_rob, 
                   robustness_denoise, max_frame_count, rad_max, max_multiplier):
    """



    Parameters
    ----------
    ref_img : Array[imsize_y, imsize_x]
        The reference image
    covs : device array[grey_imsize_y, grey_imsize_x, 2, 2]
        covariance matrices sampled at the center of each grey pixel.
    bayer_mode : bool
        Whether the burst is raw or grey
    iso_kernel : bool
        Whether isotropic kernels should be used, or handhled's kernels.
    scale : float
        scaling factor
    CFA_pattern : Array[2, 2]
        CFA pattern of the burst
    output_img : Array[SCALE*imsize_y, SCALE_imsize_x]
        The empty output image

    Returns
    -------
    None.

    """

    output_pixel_idx, output_pixel_idy = cuda.grid(2)
    output_size_y, output_size_x, _ = num.shape
    input_size_y, input_size_x = ref_img.shape
    
    if not (0 <= output_pixel_idx < output_size_x and
            0 <= output_pixel_idy < output_size_y):
        return
    
    
    if bayer_mode:
        n_channels = 3
        acc = cuda.local.array(3, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        val = cuda.local.array(3, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    else:
        n_channels = 1
        acc = cuda.local.array(1, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        val = cuda.local.array(1, dtype=DEFAULT_CUDA_FLOAT_TYPE)

    # Copying CFA locally. We will read that 9 times, so it's worth it
    # TODO threads could cooperate to read that
    local_CFA = cuda.local.array((2,2), uint8)
    for i in range(2):
        for j in range(2):
            local_CFA[i,j] = uint8(CFA_pattern[i,j])
    
    
    coarse_ref_sub_pos = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE) # y, x
    coarse_ref_sub_pos[0] = output_pixel_idy / scale          
    coarse_ref_sub_pos[1] = output_pixel_idx / scale
    
    for chan in range(n_channels):
        acc[chan] = 0
        val[chan] = 0

    
    # computing kernel
    # TODO this is rather slow and could probably be sped up
    if not iso_kernel:
        interpolated_cov = cuda.local.array((2, 2), dtype = DEFAULT_CUDA_FLOAT_TYPE)
        cov_i = cuda.local.array((2, 2), dtype=DEFAULT_CUDA_FLOAT_TYPE)
        
        
        # fetching the 4 closest covs
        close_covs = cuda.local.array((2, 2, 2 ,2), DEFAULT_CUDA_FLOAT_TYPE)
        grey_pos = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)
        if bayer_mode:
            grey_pos[0] = (coarse_ref_sub_pos[0]-0.5)/2 # grey grid is offseted and twice more sparse
            grey_pos[1] = (coarse_ref_sub_pos[1]-0.5)/2
            
        else:
            grey_pos[0] = coarse_ref_sub_pos[0] # grey grid is exactly the coarse grid
            grey_pos[1] = coarse_ref_sub_pos[1]
    
    
        # clipping the coordinates to stay in bound
        floor_x = int(max(math.floor(grey_pos[1]), 0))
        floor_y = int(max(math.floor(grey_pos[0]), 0))
        
        ceil_x = min(floor_x + 1, covs.shape[1]-1)
        ceil_y = min(floor_y + 1, covs.shape[0]-1)
        for i in range(0, 2):
            for j in range(0, 2):
                close_covs[0, 0, i, j] = covs[floor_y, floor_x,
                                              i, j]
                close_covs[0, 1, i, j] = covs[floor_y, ceil_x,
                                              i, j]
                close_covs[1, 0, i, j] = covs[ceil_y, floor_x,
                                              i, j]
                close_covs[1, 1, i, j] = covs[ceil_y, ceil_x,
                                              i, j]

        # interpolating covs
        interpolate_cov(close_covs, grey_pos, interpolated_cov)
        
        if abs(interpolated_cov[0, 0]*interpolated_cov[1, 1] - interpolated_cov[0, 1]*interpolated_cov[1, 0]) > EPSILON_DIV: # checking if cov is invertible
            invert_2x2(interpolated_cov, cov_i)

        else: # if not invertible, identity matrix
            cov_i[0, 0] = 1
            cov_i[0, 1] = 0
            cov_i[1, 0] = 0
            cov_i[1, 1] = 1
            
    
    # fetching acc robustness if required
    # The robustness of the center of the patch is picked through neirest neigbhoor interpolation
    if robustness_denoise : 
        local_acc_r = acc_rob[min(round(grey_pos[0]), acc_rob.shape[0]-1),
                              min(round(grey_pos[1]), acc_rob.shape[1]-1)]
        
        additional_denoise_power = denoise_power_merge(local_acc_r, max_multiplier, max_frame_count)
        rad = denoise_range_merge(local_acc_r, rad_max, max_frame_count)
        
    else:
        additional_denoise_power = 1
        rad = 1     

        
    center_x = round(coarse_ref_sub_pos[1])
    center_y = round(coarse_ref_sub_pos[0])
    for i in range(-rad, rad+1):
        for j in range(-rad, rad+1):
            pixel_idx = center_x + j
            pixel_idy = center_y + i
            
            # in bound condition
            if (0 <= pixel_idx < input_size_x and
                0 <= pixel_idy < input_size_y):
            
                # checking if pixel is r, g or b
                if bayer_mode : 
                    channel = local_CFA[pixel_idy%2, pixel_idx%2]
                else:
                    channel = 0
                    
                # By fetching the value now, we can compute the kernel weight 
                # while it is called from global memory
                c = ref_img[pixel_idy, pixel_idx]
            
                # computing distance
                dist_x = pixel_idx - coarse_ref_sub_pos[1]
                dist_y = pixel_idy - coarse_ref_sub_pos[0]
            
                ### Computing w
                if iso_kernel : 
                    y = max(0, 2*(dist_x*dist_x + dist_y*dist_y))
                else:
                    y = max(0, quad_mat_prod(cov_i, dist_x, dist_y))
                    # y can be slightly negative because of numerical precision.
                    # I clamp it to not explode the error with exp
                    
                
                # this is equivalent to multiplying the covariance,
                # but at the cost of one scalar operation (instead of 4)
                y/= additional_denoise_power
                
                # TODO it works because in grey mode, the signal is reconstrcuted
                # from 4 times more points. We could make thaty cleaner
                # By picking k_detail smaller
                
                w = math.exp(-0.5*y)
                ############
                brightness = c*ref_exposure
                
                if brightness > 0.99:
                    h = 0
                else:
                    id_noise = clamp(round(1000 * brightness), 0, 1001)
                    h = ref_exposure/std_curve[id_noise] # Tsin et al.
                
                val[channel] += c*w*h
                acc[channel] += w*h
    
    if robustness_denoise and local_acc_r < max_frame_count:
        # Overwritting values to enforce single frame
        # demosaicing        
        for chan in range(n_channels):
            num[output_pixel_idy, output_pixel_idx, chan] = val[chan]
            den[output_pixel_idy, output_pixel_idx, chan] = acc[chan]
        
    else:
        for chan in range(n_channels):
            num[output_pixel_idy, output_pixel_idx, chan] += val[chan]
            den[output_pixel_idy, output_pixel_idx, chan] += acc[chan]
          
    
def merge(comp_img, alignments, covs, r, exposure,
          num, den,
          options, params):
    """
    Implementation of Alg. 4: Accumulation
    Accumulates comp_img (J_n, n>1) into num and den, based on the alignment
    V_n, the covariance matrices Omega_n and the robustness mask estimated before.
    The size of the merge_result is adjustable with params['scale']


    Parameters
    ----------
    comp_imgs : device Array [imsize_y, imsize_x]
        The non-reference image to merge (J_n)
    alignments : device Array[n_tiles_y, n_tiles_x, 2]
        The final estimation of the tiles' alignment V_n(p)
    covs : device array[imsize_y//2, imsize_x//2, 2, 2]
        covariance matrices Omega_n
    r : Device_Array[imsize_y//2, imsize_x//2]
        Robustness mask r_n
    num : device Array[s*imshape_y, s*imshape_x]
        Numerator of the accumulator
    den : device Array[s*imshape_y, s*imshape_x]
        Denominator of the accumulator
        
    options : Dict
        Options to pass
    params : Dict
        parameters

    Returns
    -------
    None

    """
    scale = params['scale']
    
    CFA_pattern = cuda.to_device(params['exif']['CFA Pattern'])
    bayer_mode = params['mode'] == 'bayer'
    iso_kernel = params['kernel'] == 'iso'
    tile_size = params['tuning']['tileSize']
    
    # moving noise model to GPU
    cuda_std_curve = cuda.to_device(params['std_curve'])

    native_im_size = comp_img.shape
    # casting to integer to account for floating scale
    output_size = (round(scale*native_im_size[0]), round(scale*native_im_size[1]))


    # dispatching threads. 1 thread for 1 output pixel
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS) # maximum, we may take less
    blockspergrid_x = math.ceil(output_size[1]/threadsperblock[1])
    blockspergrid_y = math.ceil(output_size[0]/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
                    
    accumulate[blockspergrid, threadsperblock](
        comp_img, alignments, covs, r,
        exposure,
        bayer_mode, iso_kernel, scale, tile_size, CFA_pattern, cuda_std_curve,
        num, den)



@cuda.jit
def accumulate(comp_img, alignments, covs, r,
               exposure,
               bayer_mode, iso_kernel, scale, tile_size, CFA_pattern, std_curve,
               num, den):
    """



    Parameters
    ----------
    comp_imgs : Array[imsize_y, imsize_x]
        The compared image
    alignements : Array[n_tiles_y, n_tiles_x, 2]
        The alignemnt vectors for each tile of the image
    covs : device array[imsize_y/2, imsize_x/2, 2, 2]
        covariance matrices sampled at the center of each bayer quad.
    r : Device_Array[imsize_y/2, imsize_x/2, 3]
            Robustness of the moving images
    bayer_mode : bool
        Whether the burst is raw or grey
    iso_kernel : bool
        Whether isotropic kernels should be used, or handhled's kernels.
    scale : float
        scaling factor
    tile_size : int
        tile size used for alignment (on the raw scale !)
    CFA_pattern : device Array[2, 2]
        CFA pattern of the burst
    output_img : Array[SCALE*imsize_y, SCALE_imsize_x]
        The empty output image

    Returns
    -------
    None.

    """

    output_pixel_idx, output_pixel_idy = cuda.grid(2)

    
    output_size_y, output_size_x, _ = num.shape
    input_size_y, input_size_x = comp_img.shape
    
    if not (0 <= output_pixel_idx < output_size_x and
            0 <= output_pixel_idy < output_size_y):
        return
    
    if bayer_mode:
        n_channels = 3
        acc = cuda.local.array(3, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        val = cuda.local.array(3, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    else:
        n_channels = 1
        acc = cuda.local.array(1, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        val = cuda.local.array(1, dtype=DEFAULT_CUDA_FLOAT_TYPE)

    # Copying CFA locally. We will read that 9 times, so it's worth it
    # TODO threads could cooperate to read that
    local_CFA = cuda.local.array((2,2), uint8)
    for i in range(2):
        for j in range(2):
            local_CFA[i,j] = uint8(CFA_pattern[i,j])

    
    coarse_ref_sub_pos = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE) # y, x
    
    coarse_ref_sub_pos[0] = output_pixel_idy / scale          
    coarse_ref_sub_pos[1] = output_pixel_idx / scale

    # fetch of the flow, as early as possible
    local_optical_flow = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    patch_idy = int(coarse_ref_sub_pos[0]//tile_size)
    patch_idx = int(coarse_ref_sub_pos[1]//tile_size)
    local_optical_flow[0] = alignments[patch_idy, patch_idx, 0]
    local_optical_flow[1] = alignments[patch_idy, patch_idx, 1]
    
    for chan in range(n_channels):
        acc[chan] = 0
        val[chan] = 0
    
    patch_center_pos = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE) # y, x


    # fetching robustness
    # The robustness of the center of the patch is picked through neirest neigbhoor interpolation

    if bayer_mode :
        y_r = clamp(round((coarse_ref_sub_pos[0] - 0.5)/2), 0, r.shape[0])
        x_r = clamp(round((coarse_ref_sub_pos[1] - 0.5)/2), 0, r.shape[1])

    # Clamping because round(0.5)=1, meaning that in extreme case this can
    # be out of bounds
    if bayer_mode :
        y_r = clamp(round((coarse_ref_sub_pos[0] - 0.5)/2), 0, r.shape[0])
        x_r = clamp(round((coarse_ref_sub_pos[1] - 0.5)/2), 0, r.shape[1])
    else:
        y_r = clamp(round(coarse_ref_sub_pos[0]), 0, r.shape[0]-1)
        x_r = clamp(round(coarse_ref_sub_pos[1]), 0, r.shape[1]-1)
    local_r = r[y_r, x_r]
        
    patch_center_pos[1] = coarse_ref_sub_pos[1] + local_optical_flow[0]
    patch_center_pos[0] = coarse_ref_sub_pos[0] + local_optical_flow[1]

    
    # updating inbound condition
    if not (0 <= patch_center_pos[1] < input_size_x and
            0 <= patch_center_pos[0] < input_size_y):
        return
    
    # computing kernel
    if not iso_kernel:
        interpolated_cov = cuda.local.array((2, 2), dtype = DEFAULT_CUDA_FLOAT_TYPE)
        cov_i = cuda.local.array((2, 2), dtype=DEFAULT_CUDA_FLOAT_TYPE)
        # fetching the 4 closest covs
        close_covs = cuda.local.array((2, 2, 2 ,2), DEFAULT_CUDA_FLOAT_TYPE)
        grey_pos = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)
        
        if bayer_mode :
            grey_pos[0] = (patch_center_pos[0]-0.5)/2 # grey grid is offseted and twice more sparse
            grey_pos[1] = (patch_center_pos[1]-0.5)/2
            
        else:
            grey_pos[0] = patch_center_pos[0] # grey grid is exactly the coarse grid
            grey_pos[1] = patch_center_pos[1]
        
        # clipping the coordinates to stay in bound
        floor_x = int(max(math.floor(grey_pos[1]), 0))
        floor_y = int(max(math.floor(grey_pos[0]), 0))
        
        ceil_x = min(floor_x + 1, covs.shape[1]-1)
        ceil_y = min(floor_y + 1, covs.shape[0]-1)
        for i in range(0, 2):
            for j in range(0, 2):
                close_covs[0, 0, i, j] = covs[floor_y, floor_x,
                                              i, j]
                close_covs[0, 1, i, j] = covs[floor_y, ceil_x,
                                              i, j]
                close_covs[1, 0, i, j] = covs[ceil_y, floor_x,
                                              i, j]
                close_covs[1, 1, i, j] = covs[ceil_y, ceil_x,
                                              i, j]

        # interpolating covs at the desired spot
        interpolate_cov(close_covs, grey_pos, interpolated_cov)

        if abs(interpolated_cov[0, 0]*interpolated_cov[1, 1] - interpolated_cov[0, 1]*interpolated_cov[1, 0]) > EPSILON_DIV: # checking if cov is invertible
            invert_2x2(interpolated_cov, cov_i)

        else: # if not invertible, identity matrix
            cov_i[0, 0] = 1
            cov_i[0, 1] = 0
            cov_i[1, 0] = 0
            cov_i[1, 1] = 1
    
    
    center_x = round(patch_center_pos[1])
    center_y = round(patch_center_pos[0])
    for i in range(-1, 2):
        for j in range(-1, 2):
            pixel_idx = center_x + j
            pixel_idy = center_y + i
            
            # in bound condition
            if (0 <= pixel_idx < input_size_x and
                0 <= pixel_idy < input_size_y):
            
                # checking if pixel is r, g or b
                if bayer_mode : 
                    channel = local_CFA[pixel_idy%2, pixel_idx%2]
                else:
                    channel = 0
                    
                # By fetching the value now, we can compute the kernel weight 
                # while it is called from global memory
                c = comp_img[pixel_idy, pixel_idx]
            
                # computing distance
                dist_x = pixel_idx - patch_center_pos[1]
                dist_y = pixel_idy - patch_center_pos[0]
            
                ### Computing w
                if iso_kernel : 
                    y = max(0, 2*(dist_x*dist_x + dist_y*dist_y))
                else:
                    y = max(0, quad_mat_prod(cov_i, dist_x, dist_y))
                    # y can be slightly negative because of numerical precision.
                    # I clamp it to not explode the error with exp
                
                # TODO it works because in grey mode, the signal is reconstrcuted
                # from 4 times more points. We could make thaty cleaner
                # By picking k_detail 4 times smaller
                w = math.exp(-0.5*y)


                ############
                brightness = c*exposure
                
                if brightness > 0.99:
                    h = 0
                else:
                    id_noise = clamp(round(1000 * brightness), 0, 1001)
                    h = exposure/std_curve[id_noise] # Tsin et al.
                    
                val[channel] += c*w*local_r*h
                acc[channel] += w*local_r*h
        
    for chan in range(n_channels):
        num[output_pixel_idy, output_pixel_idx, chan] += val[chan] 
        den[output_pixel_idy, output_pixel_idx, chan] += acc[chan]
    
    
    
    