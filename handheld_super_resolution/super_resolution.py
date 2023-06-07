# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:56:22 2022

This script contains : 
    - The implementation of Alg. 1, the main the body of the method
    - The implementation of Alg. 2, where the function necessary to
        compute the optical flow are called
    - All the operations necessary before its call, such as whitebalance,
        exif reading, and manipulations of the user's parameters.
    - The call to post-processing operations (if enabled)


@author: jamyl
"""

import os
import time

from pathlib import Path
import numpy as np
from numba import cuda


from .utils import getTime, DEFAULT_NUMPY_FLOAT_TYPE, divide, add, round_iso, DEFAULT_TORCH_FLOAT_TYPE
from .utils_image import frame_count_denoising_gauss, frame_count_denoising_median, compute_sat_mask, base_detail_decomp
from .merge import merge, merge_ref, merge_bases
from .kernels import estimate_kernels
from .robustness import init_robustness, compute_robustness
from .params import check_params_validity, get_params, merge_params
from .optical_flow.alignment import align, init_alignment


NOISE_MODEL_PATH = Path(os.getcwd()) / 'data' 
        

def main(ref_img, comp_imgs,
         ref_exposure, exposures,
         options, params):
    """
    This is the implementation of Alg. 1: HandheldBurstSuperResolution.
    Some part of Alg. 2: Registration are also integrated for optimisation.

    Parameters
    ----------
    ref_img : Array[imshape_y, imshape_x]
        Reference frame J_1
    comp_imgs : Array[N-1, imshape_y, imshape_x]
        Remaining frames of the burst J_2, ..., J_N
        
    options : dict
        verbose options.
    params : dict
        paramters.

    Returns
    -------
    num : device Array[imshape_y*s, imshape_y*s, 3]
        generated RGB image WITHOUT any post-processing.
    debug_dict : dict
        Contains (if debugging is enabled) some debugging infos.

    """
    verbose = options['verbose'] >= 1
    verbose_2 = options['verbose'] >= 2
    verbose_3 = options['verbose'] >= 3
    
    sigma_bd=1 # std for base detail decomp
    
    debug_mode = params['debug']
    debug_dict = {"robustness":[],
                  "pre alignment":[],
                  "flow":[]}
    
    accumulate_r = params['accumulated robustness denoiser']['on'] 
    base_detail = params["base detail"]
    
    #___ Moving to GPU
    cuda_ref_img = cuda.to_device(ref_img)
    cuda.synchronize()
    
    if verbose :
        print("\nProcessing reference image ---------\n")
        t1 = time.perf_counter()
    
    #___ Saturation mask
    ref_sat_mask = compute_sat_mask(cuda_ref_img, ref_exposure)
    
    
    cuda_ref_grey = cuda_ref_img
        
    align_args = init_alignment(cuda_ref_grey, ref_sat_mask, options, params)
    
    
    #___ Local stats estimation
    if verbose_2:
        cuda.synchronize()
        current_time = time.perf_counter()
        print("\nEstimating ref image local stats")
        
    ref_local_stats = init_robustness(cuda_ref_img,options, params['robustness'])
    
    if accumulate_r:
        accumulated_r = cuda.to_device(np.zeros(ref_local_stats.shape[:2]))
    else:
        accumulated_r = None

    
    
    if verbose_2 :
        cuda.synchronize()
        current_time = getTime(current_time, 'Local stats estimated (Total)')

    # zeros init of num and den
    scale = params["scale"]
    native_imshape_y, native_imshape_x = cuda_ref_img.shape
    output_size = (round(scale*native_imshape_y), round(scale*native_imshape_x))
    num = cuda.to_device(np.zeros(output_size+(3,), dtype = DEFAULT_NUMPY_FLOAT_TYPE))
    den = cuda.to_device(np.zeros(output_size+(3,), dtype = DEFAULT_NUMPY_FLOAT_TYPE))
    
    if verbose :
        cuda.synchronize()
        getTime(t1, '\nRef Img processed (Total)')
    

    n_images = comp_imgs.shape[0]
    for im_id in range(n_images):
        if verbose :
            cuda.synchronize()
            print("\nProcessing image {} ---------\n".format(im_id+1))
            im_time = time.perf_counter()
        
        exposure = exposures[im_id]
        
        #___ Moving to GPU
        cuda_img = cuda.to_device(comp_imgs[im_id])
        if verbose_3 : 
            cuda.synchronize()
            current_time = getTime(im_time, 'Arrays moved to GPU')
            
        #___ base detail decomposition
        if base_detail:
            base, detail = base_detail_decomp(cuda_img, sigma_bd)
            
        #___ Saturation mask
        sat_mask = compute_sat_mask(cuda_img, exposure)
        
        cuda_im_grey = cuda_img
        
        cuda_final_alignment = align(cuda_ref_grey, cuda_im_grey,
                                     ref_sat_mask, sat_mask,
                                     align_args, debug_dict, options, params)
        
            
        #___ Robustness
        if verbose_2 :
            cuda.synchronize()
            current_time = time.perf_counter()
            print('\nEstimating robustness')
            
        
        cuda_robustness = compute_robustness(cuda_img, ref_local_stats,
                                             ref_exposure, exposure,
                                             cuda_final_alignment,
                                             options, params['robustness'])
        if accumulate_r:
            add(accumulated_r, cuda_robustness)
        
        if verbose_2 :
            cuda.synchronize()
            current_time = getTime(current_time, 'Robustness estimated (Total)')

            
        #___ Kernel estimation
        if verbose_2 :
            cuda.synchronize()
            current_time = time.perf_counter()
            print('\nEstimating kernels')
            
        cuda_kernels = estimate_kernels(cuda_img, options, params['merging'])
        
        if verbose_2 :
            cuda.synchronize()
            current_time = getTime(current_time, 'Kernels estimated (Total)')
            
            
        #___ Merging
        if verbose_2 : 
            current_time = time.perf_counter()
            print('\nAccumulating Image')
            
        if base_detail:
            merge(detail, cuda_final_alignment, cuda_kernels, cuda_robustness, exposure,
                  num, den,
                  options, params['merging'])
        else:
            merge(cuda_im_grey, cuda_final_alignment, cuda_kernels, cuda_robustness, exposure,
                  num, den,
                  options, params['merging'])
        
        if verbose_2 :
            cuda.synchronize()
            current_time = getTime(current_time, 'Image accumulated (Total)')
        if verbose :
            cuda.synchronize()
            getTime(im_time, '\nImage processed (Total)')
            
        if debug_mode : 
            debug_dict['robustness'].append(cuda_robustness.copy_to_host())
    
    #___ Ref kernel estimation
    if verbose_2 : 
        cuda.synchronize()
        current_time = time.perf_counter()
        print('\nEstimating kernels')
        
    cuda_kernels = estimate_kernels(cuda_ref_img, options, params['merging'])
    
    if verbose_2 : 
        cuda.synchronize()
        current_time = getTime(current_time, 'Kernels estimated (Total)')
    
    #___ Merge ref
    if verbose_2 :
        cuda.synchronize()
        print('\nAccumulating ref Img')
    
    if base_detail:
        #___ base detail decomposition
        SR_base, ref_detail = merge_bases(ref_img, comp_imgs,
                                          ref_exposure, exposures,
                                          sigma_bd=sigma_bd)
        

        merge_ref(ref_detail, cuda_kernels,
                      ref_exposure,
                      num, den,
                      options, params["merging"], accumulated_r)

            
        # num is outwritten into num/den
        divide(num, den)
        # adding merged details to ref base
        add(SR_base, num[:, :, 0])
        out = SR_base
    
    else:
        merge_ref(cuda_ref_grey, cuda_kernels,
                  ref_exposure,
                  num, den,
                  options, params["merging"], accumulated_r)
        # num is outwritten into num/den
        divide(num, den)
        
        out = num[:, :, 0]
        
    
    if verbose_2 : 
        cuda.synchronize()
        getTime(current_time, 'Ref Img accumulated (Total)')
        
    if verbose_2 :
        print('\n------------------------')
        cuda.synchronize()
        current_time = getTime(current_time, 'Image normalized (Total)')
    
    if verbose :
        print('\nTotal ellapsed time : ', time.perf_counter() - t1)
    
    if accumulate_r :
        debug_dict['accumulated robustness'] = accumulated_r
        
    
    return out, debug_dict

    
    
def process(burst, exposures=None, options=None, custom_params=None):
    """
    Processes the burst of .npy grey images 

    Parameters
    ----------
    burst : np array [b, ny, nex]
        Burst of grey Images. Must be int, or float normalised in [0, 1]
    exposures : np array [b]
        List of the exposure ratios (if the burst is multi exposed)
    options : dict
        
    params : Parameters
        See params.py for more details.

    Returns
    -------
    Array
        The processed image

    """
    if options is None:
        options = {'verbose' : 0}
    currentTime, verbose_1, verbose_2 = (time.perf_counter(),
                                         options['verbose'] >= 1,
                                         options['verbose'] >= 2)
    params = {}
    multi_exposed = exposures is not None 
    
    ####
    # sat = burst > 0.95*4095
    # sat_levels = np.sum(sat, axis = (1, 2))/(sat.shape[1] * sat.shape[2])
    
    # sorted_id = np.argsort(sat_levels)
    # sat_levels = sat_levels[sorted_id]
    # burst = burst[sorted_id]
    # exposures = exposures[sorted_id]
    
    # ref_id = np.where(sat_levels < 0.01)[0][-1] # highest saturation rate under 1%
    
    ref_id = 0
    
    
    # ISO = 100/100
    # alpha = 1.80710882e-4
    # beta = 3.1937599182128e-6
    
    # means = np.mean(burst, axis = (1, 2))
    # sigmas = np.sqrt(alpha*ISO*means + beta*ISO**2)
    # SNR = means/sigmas
    
    ####
    
    if multi_exposed:
        ref_exposure = exposures[ref_id]
        exposures = np.delete(exposures, ref_id, axis=0)
    else:
        ref_exposure = 1
        exposures = np.ones(burst.shape[0]-1)
        
    ref_raw = burst[ref_id]
    raw_comp = np.delete(burst, ref_id, axis=0)
    
    ISO = 100
    
    
    # Packing noise model related to picture ISO
    std_noise_model_label = 'noise_model_std_ISO_{}'.format(ISO)
    diff_noise_model_label = 'noise_model_diff_ISO_{}'.format(ISO)
    std_noise_model_path = (NOISE_MODEL_PATH / std_noise_model_label).with_suffix('.npy')
    diff_noise_model_path = (NOISE_MODEL_PATH / diff_noise_model_label).with_suffix('.npy')
    
    std_curve = np.load(std_noise_model_path)
    diff_curve = np.load(diff_noise_model_path)
    
    
    if verbose_2:
        currentTime = getTime(currentTime, ' -- Read .npy files')

    if not np.issubdtype(type(ref_raw[0,0]), np.integer):
        if not ((0 <= ref_raw.all()) and  (ref_raw.all() <= 1)):
            raise ValueError('Image burst is float type but some values are \
                             outside of the [0, 1] range.')

    
    if np.issubdtype(type(ref_raw[0,0]), np.integer):
        ## Here do black and white level correction and white balance processing for all image in comp_images
        ## Each image in comp_images should be between 0 and 1.
        ## ref_raw is a (H,W) array
        ref_raw = ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE)
        ref_raw /= 4095 # not true for all data ! Adapt it to the data
        
        ref_raw = np.clip(ref_raw, 0.0, 1.0)
        
    if np.issubdtype(type(raw_comp[0,0,0]), np.integer):
        raw_comp = raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE)
        raw_comp /= 4095
        raw_comp = np.clip(raw_comp, 0., 1.)
    
    #___ normalisation with exposure time
    if multi_exposed:
        ref_raw /= ref_exposure
        raw_comp /= exposures[:, None, None]    
    
    
    #___ Estimating ref image SNR
    brightness = np.mean(ref_raw)
    
    id_noise = round(1000*brightness)
    std = std_curve[id_noise]
    
    SNR = brightness/std
    # TODO check real camera data
    # SNR = 35
    if verbose_1:
        print(" ", 10*"-")
        print('|ISO : {}'.format(ISO))
        print('|Image brightness : {:.2f}'.format(brightness))
        print('|expected noise std : {:.2e}'.format(std))
        print('|Estimated SNR : {:.2f}'.format(SNR))
    
    SNR_params = get_params(SNR)
    
    #__ Merging params dictionnaries
    
    # checking (just in case !)
    # check_params_validity(SNR_params, ref_raw.shape)
    
    if custom_params is not None :
        params = merge_params(dominant=custom_params, recessive=SNR_params)
        check_params_validity(params, ref_raw.shape)
        
    #__ adding metadatas to dict 
    if not 'noise' in params['merging'].keys(): 
        params['merging']['noise'] = {}

    # is the algorithm had to be run on a specific sensor,
    # the precise values of alpha and beta could be used instead
    if params['mode'] == 'grey':
        # Values for the L1B sensor. dividing to express noise model as a function of I normalised in [0, 1] 
        alpha = 0.119/4095    
        beta = 12.050/(4095**2)
    else:
        raise ValueError('.npy should be run in grey mode')
        
    params['merging']['noise']['alpha'] = alpha
    params['merging']['noise']['beta'] = beta
    params['merging']['noise']['ISO'] = ISO
    
    ## Writing exifs data into parameters
    if not 'exif' in params['merging'].keys(): 
        params['merging']['exif'] = {}
    if not 'exif' in params['robustness'].keys(): 
        params['robustness']['exif'] = {}
    
    params['robustness']['exif']['CFA Pattern'] = np.zeros((2,2))
    params['merging']['exif']['CFA Pattern'] = np.zeros((2,2))
    
    params['ISO'] = ISO
    
    params['robustness']['std_curve'] = std_curve
    params['merging']['std_curve'] = std_curve
    params['robustness']['diff_curve'] = diff_curve
    
    # copying parameters values in sub-dictionaries
    if 'scale' not in params["merging"].keys() :
        params["merging"]["scale"] = params["scale"]
    if 'scale' not in params['accumulated robustness denoiser'].keys() :
        params['accumulated robustness denoiser']["scale"] = params["scale"]
    if 'tileSize' not in params["kanade"]["tuning"].keys():
        params["kanade"]["tuning"]['tileSize'] = params['block matching']['tuning']['tileSizes'][0]
    if 'tileSize' not in params["robustness"]["tuning"].keys():
        params["robustness"]["tuning"]['tileSize'] = params['kanade']['tuning']['tileSize']
    if 'tileSize' not in params["merging"]["tuning"].keys():
        params["merging"]["tuning"]['tileSize'] = params['kanade']['tuning']['tileSize']


    if 'mode' not in params["kanade"].keys():
        params["kanade"]["mode"] = params['mode']
    if 'mode' not in params["robustness"].keys():
        params["robustness"]["mode"] = params['mode']
    if 'mode' not in params["merging"].keys():
        params["merging"]["mode"] = params['mode']
    if 'mode' not in params['accumulated robustness denoiser'].keys():
        params['accumulated robustness denoiser']["mode"] = params['mode']
    
    # deactivating robustness accumulation if robustness is disabled
    params['accumulated robustness denoiser']['median']['on'] &= params['robustness']['on']
    params['accumulated robustness denoiser']['gauss']['on'] &= params['robustness']['on']
    params['accumulated robustness denoiser']['merge']['on'] &= params['robustness']['on']
    
    params['accumulated robustness denoiser']['on'] = \
        (params['accumulated robustness denoiser']['gauss']['on'] or
         params['accumulated robustness denoiser']['median']['on'] or
         params['accumulated robustness denoiser']['merge']['on'])
     
    # if robustness aware denoiser is in merge mode, copy in merge params
    if params['accumulated robustness denoiser']['merge']['on']:
        params['merging']['accumulated robustness denoiser'] = params['accumulated robustness denoiser']['merge']
    else:
        params['merging']['accumulated robustness denoiser'] = {'on' : False}
    
    
        
        
    
    
    #___ Running the handheld pipeline
    handheld_output, debug_dict = main(ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE), raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE),
                                       ref_exposure, exposures, 
                                       options, params)
    
    
    
    #___ Performing frame count aware denoising if enabled
    median_params = params['accumulated robustness denoiser']['median']
    gauss_params = params['accumulated robustness denoiser']['gauss']
    
    median = median_params['on']
    gauss = gauss_params['on']
    post_frame_count_denoise = (median or gauss)
    
    if post_frame_count_denoise : 
        if verbose_2:
            print('-- Robustness aware bluring')
        
        if median:
            handheld_output = frame_count_denoising_median(handheld_output, debug_dict['accumulated robustness'],
                                                           median_params)
        if gauss:
            handheld_output = frame_count_denoising_gauss(handheld_output, debug_dict['accumulated robustness'],
                                                          gauss_params)


    #___ post processing
    params_pp = params['post processing']
    post_processing_enabled = params_pp['on']
    
    if post_processing_enabled:
        raise NotImplementedError("no post processing for .npy images")

    else:
        output_image = handheld_output.copy_to_host()
        
    #__ return
    
    if params['debug']:
        return output_image, debug_dict
    else:
        return output_image
    
