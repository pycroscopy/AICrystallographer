import numpy as np
import matplotlib.pyplot as plt
import dcnn
from atomfind import *


plot_results = 1 # 1 - make a plot, 0 - no plot

# Folder and filename for the saved weights
weights_folder = './saved_models/'
weights_file = 'G-test-4-1-best_weights.pt'

# Folder and filename for the experimental image
image_folder = 'exp_data/'
image_file = 'G-Si-2nm.npy'

# Load model skeleton
model = dcnn.atomsegnet()
# Load trained weights
model = dcnn.load_torchmodel(weights_folder+weights_file, model)

#Load image as numpy array
imgdata = np.load(image_folder+image_file)

# Apply a trained model to the loaded data
img, dec = dl_image(imgdata, model).decode()

# Get atomic coordinates:
coord = find_atoms(dec).get_all_coordinates()

# Save the results
np.save(image_folder+image_file+'-dec.npy', img)
np.save(image_folder+image_file+'-coord.npy', coord)
print('Neural network output and atomic coordinates saved to disk')

# Plot results ( for image stack (movie), plots the first image (frame) )
if plot_results == 1:
    k = 0
    y, x,_ = coord[0].T
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(img[k,:,:,0], cmap = 'gray')
    ax1.set_title('Experimental')
    ax2.imshow(dec[k,:,:,0], cmap = 'jet', Interpolation = 'Gaussian')
    ax2.set_title('Neural Network Output')
    ax3.imshow(img[k,:,:,0], cmap = 'gray')
    ax3.scatter(x, y, s = 1, c = 'red')
    ax3.set_title('Coordinates')
    for ax in [ax1, ax2, ax3]:
        ax.axis('off')
    plt.show()
