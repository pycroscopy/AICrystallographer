import sys
import dcnn
from atomfind import *
from utils import *
import graphs
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", module="networkx")

image_file = sys.argv[1]

# Folder and filename for the saved weights
weights_folder = './saved_models/'
weights_file = 'G-Si-m-2-3-best_weights.pt'
# Load model skeleton
model = dcnn.atomsegnet(nb_classes=3)
# Load trained weights
model = dcnn.load_torchmodel(weights_folder+weights_file, model)
#Load image as numpy array
imgdata, metadata = open_hdf(image_file)
img_size = metadata['scan size']
imgdata = optimize_image_size(imgdata, img_size)
# Apply a trained model to the loaded data
img, dec = dl_image(imgdata, model, use_gpu=False, nb_classes=3).decode()
# Get atomic coordinates:
coord = find_atoms(dec).get_all_coordinates()
atoms, approx_max_bonds = atom_bond_dict()
# plot results
y, x,_ = coord[0].T
fig, ax = plt.subplots(1, 3)
ax[0].imshow(img[0, :, :, 0], cmap='gray')
ax[0].set_title('Experimental')
ax[1].imshow(dec[0, :, :, :], cmap='jet', Interpolation='Gaussian')
ax[1].set_title('Neural Network Output')
ax[2].imshow(img[0, :, :, 0], cmap='gray')
ax[2].scatter(x, y, s=1, c='red')
ax[2].set_title('Coordinates')
for _ax in fig.axes:
    _ax.axis('off')
plt.show(block=False)
 # create and plot graphs
graphs.construct_graphs(img, img_size, coord[0], atoms, approx_max_bonds, image_file)
plt.show()
