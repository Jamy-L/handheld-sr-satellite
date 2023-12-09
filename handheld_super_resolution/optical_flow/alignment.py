# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:46:51 2023

@author: jamyl
"""
from numba import cuda
import time
from .block_matching import init_block_matching, align_image_block_matching
from ..utils import getTime
from .ICA import ICA_optical_flow, init_ICA
from .aligners import phase_correlation_alignment, farneback_alignment, dense_ica_affinity_alignment, global_ica_alignment
from .deep_flow import get_deep_flow, th_get_deep_flow


def init_alignment(cuda_ref_grey, ref_sat_mask, options, params):
    verbose_2 = options['verbose'] >= 2
    
    if params['alignment'] == 'patch ICA':
        #___ Block Matching
        if verbose_2 :
            cuda.synchronize()
            current_time = time.perf_counter()
            print('\nBeginning Block Matching initialisation')
            
        reference_pyramid = init_block_matching(cuda_ref_grey, options, params['block matching'])
        
        if verbose_2 :
            cuda.synchronize()
            current_time = getTime(current_time, 'Block Matching initialised (Total)')
        
        #___ ICA : compute grad and hessian    
        if verbose_2 :
            cuda.synchronize()
            current_time = time.perf_counter()
            print('\nBeginning ICA initialisation')
            
        ref_gradx, ref_grady, hessian = init_ICA(cuda_ref_grey, options, params['kanade'], ref_sat_mask)
        
        if verbose_2 :
            cuda.synchronize()
            current_time = getTime(current_time, 'ICA initialised (Total)')
            
        align_args = {'reference_pyramid':reference_pyramid,
                      'ref_gradx':ref_gradx,
                      'ref_grady':ref_grady,
                      'hessian':hessian}
        
        return align_args
    else:
        return

def align(cuda_ref_grey, cuda_im_grey, ref_sat_mask, sat_mask, align_args, debug_dict, options, params):
    verbose_2 = options['verbose'] >= 2
    debug_mode = params['debug']
    
    
# ---- BM + ICA homemade
    if params['alignment'] == 'patch ICA':
        reference_pyramid = align_args['reference_pyramid']
        ref_gradx = align_args['ref_gradx']
        ref_grady = align_args['ref_grady']
        hessian = align_args['hessian']
        
        
        
    #___ Block Matching
        if verbose_2 :
            cuda.synchronize()
            current_time = time.perf_counter()
            print('Beginning block matching')
        
        pre_alignment = align_image_block_matching(cuda_im_grey, reference_pyramid, options, params['block matching'])
        
        if debug_mode:
            debug_dict["pre alignment"].append(pre_alignment.copy_to_host())
        
        if verbose_2 :
            cuda.synchronize()
            current_time = getTime(current_time, 'Block Matching (Total)')
            
            
        #___ ICA
        if verbose_2 :
            cuda.synchronize()
            current_time = time.perf_counter()
            print('\nBeginning ICA alignment')
        
        cuda_final_alignment = ICA_optical_flow(
            cuda_im_grey, cuda_ref_grey, ref_gradx, ref_grady, hessian, pre_alignment, options, params['kanade'], ref_sat_mask, sat_mask)
        
        if debug_mode:
            debug_dict["flow"].append(cuda_final_alignment.copy_to_host())
            
        if verbose_2 : 
            cuda.synchronize()
            current_time = getTime(current_time, 'Image aligned using ICA (Total)')

# ---- Phase correlation

    # cuda_final_alignment = phase_correlation_alignment(cuda_ref_grey, cuda_im_grey,
    #                                                     tile_size=params['kanade']['tuning']['tileSize'])
    # if debug_mode:
    #     debug_dict["flow"].append(cuda_final_alignment.copy_to_host())
            
# ---- Farneback

    # cuda_final_alignment = farneback_alignment(cuda_ref_grey, cuda_im_grey,
    #                                             tile_size=params['kanade']['tuning']['tileSize'])
    
    # if debug_mode:
    #     debug_dict["flow"].append(cuda_final_alignment.copy_to_host())
            
# ---- Global affinity ICA

    # cuda_final_alignment = dense_ica_affinity_alignment(cuda_ref_grey, cuda_im_grey, tile_size=params['kanade']['tuning']['tileSize'])

    # if debug_mode:
    #     debug_dict["flow"].append(cuda_final_alignment.copy_to_host())
        
    if params['alignment'] == 'ICA':
# ---- Global translation ICA

        cuda_final_alignment = global_ica_alignment(cuda_ref_grey, cuda_im_grey, tile_size=params['kanade']['tuning']['tileSize'])

        if debug_mode:
            debug_dict["flow"].append(cuda_final_alignment.copy_to_host())

    if params['alignment'] == 'Fnet':
#---- Flow net
        cuda_final_alignment = get_deep_flow(cuda_ref_grey, cuda_im_grey, params["kanade"]["tuning"]['tileSize'])
        
        
    return cuda_final_alignment
