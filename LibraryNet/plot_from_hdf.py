import sys
import matplotlib.pyplot as plt
from utils import *

imgdata, _ = open_hdf(sys.argv[1])
plt.imshow(imgdata, cmap='gray')
plt.show()
