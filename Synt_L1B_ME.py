# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:53:43 2023

@author: jamyl
"""

import numpy as np
from pathlib import Path

from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from scipy import ndimage

sigma_blur = 0.3

burst_size = 10


L1B_original_path = Path('path/to/L1B.npy')
L1B_synt_path = Path('path/to/generated_dataset.py') # where the data will be created


L1B_original = np.load(L1B_original_path)/4095
L1B_blurred = np.empty_like(L1B_original)
n_images, *HR_imshape = L1B_original.shape


#%% Blurring

for i in tqdm(range(n_images), desc= "Blurring"):
    L1B_blurred[i] = gaussian_filter(L1B_original[i], sigma=sigma_blur)

# np.save(L1B_synt_path/'blured.npy', L1B_blurred)

#%% Picking warps

def pick_warps(n_warps, R_max=2):
    warps = R_max * (np.random.random((n_warps, 2)) - 0.5) #between -R_max and R_max
    while True:
        norms = np.abs(warps[:, 0]) + np.abs(warps[:, 1])
        invalid = warps[norms > R_max]
        n_invalid = invalid.shape[0]
        if n_invalid == 0:
            break
        else:
            invalid = R_max * (np.random.random((n_invalid, 2)) - 0.5)
    
    return warps    

    
warps = np.empty((n_images, burst_size-1, 2))
for i in tqdm(range(n_images), desc='picking warps'):
    warps[i] = pick_warps(burst_size-1)


def shift(image, warps):
    image = np.pad(image, ((64, 64), (64, 64)), mode='reflect')
    fft = np.fft.fftshift(np.fft.fft2(image))[None, :, :]
    Ny, Nx = image.shape
    FX_grid = np.arange(Nx)
    FY_grid = np.arange(Ny)
    
    Fx, Fy = np.meshgrid(FX_grid, FY_grid)
    Fx = Fx[None]
    Fy = Fy[None] #adding channel
    
    
    fft = fft * np.exp(-2j*np.pi*(warps[:, 0, None, None]*Fx/Nx + warps[:, 1, None, None]*Fy/Ny))
    
    output = np.fft.ifft2(np.fft.ifftshift(fft))
    output = np.abs(output.real)[:, 64:-64, 64:-64]
    return output

HR_warped = np.empty((n_images, burst_size, *HR_imshape))
for i in tqdm(range(n_images), desc= "Warping"):
    HR_warped[i, 0] = L1B_blurred[i]
    
    # HR_warped[i, 1:] = shift(L1B_blurred[i], warps[i])
    for j in range(warps.shape[1]):
        shifted = ndimage.shift(L1B_blurred[i], warps[i, j])
        HR_warped[i, j+1] = shifted * np.mean(L1B_blurred[i])/np.mean(shifted)
        
HR_warped = np.clip(HR_warped, 0, 1)
# np.save(L1B_synt_path/'shifted.npy', HR_warped)

#%% Downsampling
print('Downsampling')
downsampled = HR_warped[:, :, ::2, ::2]
    # output = np.clip(output*np.mean(image)/np.mean(output, axis=(1, 2), keepdims=True), 0, 1)
# np.save(L1B_synt_path/'downsampled.npy', downsampled)


#%% Picking exposure ratios
print('Picking ratios')

alphas = np.random.random((n_images, burst_size-1))*(1.4 - 1.2) + 1.2 # random in [1.2, 1.4]
choices = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
c = np.random.choice(choices, size = (n_images, burst_size-1))
ratios = alphas**c
ratios = np.concatenate((np.ones(n_images)[:, None], ratios), axis=1) # adding ratio 1 for ref

#%% Applying noise

normalised_noises = np.random.normal(0, 1, size=downsampled.shape)

a = 0.26
b = 27

noise_std = np.sqrt(a * ratios[:, :, None, None] * downsampled * 4095 + b) / 4095
# downsampled is normalised : denormalising it.
# Ratio is used instead of noise_ratio because the noise does not depend on the jitter.

noised = np.clip(downsampled*ratios[:, :, None, None] + noise_std * normalised_noises, 0, 1)


#%% Saturating over exposed pixels

noised = np.clip(noised, 0, 1)
np.save(L1B_synt_path/'noised.npy', noised)


#%% Saving jittered

np.save(L1B_synt_path/'ratios_gt.npy', ratios)

jitter_rates = [0, 0.05, 0.2]
print('Jittering')

for jitter_rate in tqdm(jitter_rates):
    jitter_noise = (np.random.random((n_images, burst_size-1))*2 - 1)*jitter_rate    
    
    jitter_noise = np.concatenate((np.zeros(n_images)[:, None], jitter_noise), axis=1) # adding ratio 1 for ref
    
    noised_ratios = (1 + jitter_noise)*ratios
    
    np.save(L1B_synt_path/'ratios_noised_{}.npy'.format(int(100*jitter_rate)), noised_ratios)




    




