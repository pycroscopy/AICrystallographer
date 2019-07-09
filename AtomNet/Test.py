# -*- coding: utf-8 -*-
"""
author: Maxim Ziatdinov
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import dcnn
from atomfind import *


plot_results = 1  # 1 - make a plot, 0 - no plot

# Folder and filename for the experimental image
image_folder = 'exp_data/'
image_file = 'G-Si-2nm.npy'

# Specify type of the lattice
lattice_type = 'graphene'

# Load model with pre-trained weights
model = dcnn.load_torchmodel(lattice_type)

# Load image as numpy array
imgdata = np.load(os.path.join(image_folder, image_file))

# Apply a trained model to the loaded data
img, dec = dl_image(imgdata, model).decode()

# Get atomic coordinates:
coord = find_atoms(dec).get_all_coordinates()

# Save the results
np.save(image_folder+os.path.splitext(image_file)[0]+'-dec.npy', img)
np.save(image_folder+os.path.splitext(image_file)[0]+'-coord.npy', coord)
print('Neural network output and atomic coordinates saved to disk')

# Plot results ( for image stack (movie), plots the first image (frame) )
if plot_results == 1:
    k = 0
    y, x, _ = coord[0].T
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img[k, :, :, 0], cmap='gray')
    ax[0].set_title('Experimental')
    ax[1].imshow(dec[k, :, :, 0], cmap='jet', Interpolation='Gaussian')
    ax[1].set_title('Neural Network Output')
    ax[2].imshow(img[k, :, :, 0], cmap='gray')
    ax[2].scatter(x, y, s=1, c='red')
    ax[2].set_title('Coordinates')
    for _ax in fig.axes:
        _ax.axis('off')
    plt.show()
