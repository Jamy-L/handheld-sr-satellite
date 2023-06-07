# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 18:10:18 2023

This is an example script showing how to call the pipeline and specify
the parameters.

Make sure that the example bursts have been downloaded from the release version!!


@author: jamyl
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from handheld_super_resolution import process

# Specify verbose options
options = {'verbose' : 1}



# Specify the scale as follows. All the parameters are automatically 
# choosen but can be overwritten : check params.py to see the entire list
# of configurable parameters.
params={
        "scale":2,
        'base detail' : True, # Whether to use the base detail decomposition or not
        'alignment' : 'Fnet', # 'Fnet', 'ICA', 'patch ICA', the alignment methode
        }

# Note : alignment "ICA" corresponds to globa ica and require to install
# the "icaflow" module

# calling the pipeline
burst_path = 'P:/L1B_synthetic/stored/multi_exposure/noised.npy'
exposure_path = 'P:/L1B_synthetic/stored/multi_exposure/ratios_noised_20.npy'
i = 10

burst = np.load(burst_path)[i]
exposures = np.load(exposure_path)[i]

output_img = process(burst, exposures, options, params)


# saving the result
vmin=0
vmax=0.5
os.makedirs('./results', exist_ok=True)
plt.imsave('./results/output_img.png', output_img, cmap="gray", vmin=vmin, vmax=vmax)
plt.imsave('./results/ref_img.png', burst[0], cmap="gray", vmin=vmin, vmax=vmax)


# plotting the result
plt.figure("output")
plt.imshow(output_img, cmap="gray", interpolation = 'none')
plt.xticks([])
plt.yticks([])

