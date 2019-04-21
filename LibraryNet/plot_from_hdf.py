import sys
import matplotlib.pyplot as plt
from utils import *

try:
    imgdata, _ = open_hdf(sys.argv[1])
    plt.imshow(imgdata, cmap='gray')
    plt.show()
except KeyError:
    imgdata, _, coord = open_library_hdf(sys.argv[1], atom_bond_dict()[0])
    plt.imshow(imgdata[0, :, :, 0], cmap='gray')
    plt.scatter(coord[:, 1], coord[:, 0], c=coord[:, 2], s=30, cmap='RdYlGn')
    plt.show()
