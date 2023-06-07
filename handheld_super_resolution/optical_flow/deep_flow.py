# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 21:32:40 2023

@author: jamyl
"""
import numpy as np
from HDR_DSP_SR.shiftandadd import shiftAndAdd, featureAdd, featureWeight
from HDR_DSP_SR.models import EncoderNet, DecoderNet, FNet
from HDR_DSP_SR.warpingOperator import WarpedLoss, TVL1, base_detail_decomp, BlurLayer
import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur
from ..utils import DEFAULT_TORCH_FLOAT_TYPE
import os 
from pathlib import Path
from .flow_utils import patchify_dense_flow

from numba import cuda

device = "cuda"
gaussian_filter = GaussianBlur(11, sigma=1).to(device)
warping = WarpedLoss(interpolation = 'bicubicTorch') 
num_features = 64
num_blocks = 4


criterion = nn.L1Loss()
feature_mode = ['Avg', 'Max', 'Std']
nb_mode = len(feature_mode)

# train_bs, val_bs, lr_fnet, factor_fnet, patience_fnet, lr_decoder, factor_decoder, patience_decoder, lr_encoder, factor_encoder, patience_encoder, num_epochs, warp_weight, TVflow_weight = args.train_bs, args.val_bs, args.lr_fnet, args.factor_fnet, args.patience_fnet,  args.lr_decoder, args.factor_decoder, args.patience_decoder, args.lr_encoder, args.factor_encoder, args.patience_encoder, args.num_epochs, args.warp_weight, args.TVflow_weight

Decoder = DecoderNet(in_dim=1+nb_mode*num_features).float().to(device)
Encoder = EncoderNet(in_dim=2,conv_dim=64, out_dim=num_features, num_blocks=num_blocks).float().to(device)
Fnet = FNet().float().to(device)


path = Path(os.path.dirname(__file__)).parents[1] / 'HDR_DSP_SR/checkpoint.pth.tar'
checkpoint = torch.load(path, map_location=torch.device('cpu'))
Decoder.load_state_dict(checkpoint['state_dictDecoder']) 
Encoder.load_state_dict(checkpoint['state_dictEncoder']) 
Fnet.load_state_dict(checkpoint['state_dictFnet']) 

Fnet.eval()

def get_deep_flow(ref, image, tile_size):
    # numba to torch hack
    torch_ref = torch.as_tensor(ref, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
    torch_img = torch.as_tensor(image, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
    stack = torch.stack((torch_ref, torch_img), axis=0)[None]
    
    dense_flow = flowEstimation(stack*3.4, ME=Fnet, device=device)[0, 1]
    dense_flow = torch.moveaxis(dense_flow, (1, 2), (0, 1)) # moving direction channel last
     
    # dense_flow = torch.flip(dense_flow, (2,))
    dense_flow *= -1
    
    # torch to numba 
    numba_flow = cuda.as_cuda_array(dense_flow.detach())
    
    # patchify flow
    patchwise_flow = patchify_dense_flow(numba_flow, tile_size)
    return patchwise_flow

def th_get_deep_flow(th_burst):
    dense_flow = flowEstimation(th_burst*3.4, ME=Fnet, device=device)[0, 1]
    dense_flow = torch.moveaxis(dense_flow, (1, 2), (0, 1)) # moving direction channel last
    return dense_flow


def flowEstimation(samplesLR, ME, device):
    """
    Compute the optical flows from the other frames to the reference:
    samplesLR: Tensor b, num_im, h, w
    ME: Motion Estimator
    """

    b, num_im, h, w = samplesLR.shape

    samplesLRblur = gaussian_filter(samplesLR)


    samplesLR_0 = samplesLRblur[:,:1,...] #b, 1, h, w


    samplesLR_0 = samplesLR_0.repeat(1, num_im, 1,1)  #b, num_im, h, w
    samplesLR_0 = samplesLR_0.reshape(-1, h, w)
    samplesLRblur = samplesLRblur.reshape(-1, h, w)  #b*num_im, h, w
    concat = torch.cat((samplesLRblur.unsqueeze(1), samplesLR_0.unsqueeze(1)), axis = 1) #b*(num_im), 2, h, w 
    flow = ME(concat.to(device)) #b*(num_im), 2, h, w 

    flow[::num_im] = 0
    # flow doesn not have the same shape as the input for some reason
    _, _, h_, w_ = flow.shape
    return flow.reshape(b, num_im, 2, h_, w_)
    
    