Status: <b>active project</b>

Please feel free to suggest possible modifications of a workflow to better suite the microscopy/ferroelectric community needs.</p>

<h1>FerroNet</h1>
Application of different machine learning tools (neural networks, dimensionality reduction, clustering/unmixing) for analysis of ferroic distortions in high-resolution scanning transmission electron microscopy data from perovskites.</p>

## How to use
Run the analysis using the [Test notebook](https://colab.research.google.com/github/pycroscopy/AICrystallographer/blob/master/FerroNet/Test.ipynb) - just upload a file with experimental image in numpy format (it will ask for it) and run each cell top to bottom. We recommend to start with the [FerroicBlocks notebook](https://colab.research.google.com/github/pycroscopy/AICrystallographer/blob/master/FerroNet/FerroicBlocks.ipynb) and use the experimental images of LBFO from the FerroNet package to get a better understanding of how the whole approach works. As long as your images are close in size and structure of the lattice to those in the examples, everything should work fine 😊 We also plan to add a more universal network in the near future. 

The input image data should be in a numpy format. If you have a preferred method of translating your experimental data to the numpy format, please use it. You can also make use of PyUSID/Pycroscopy translators. Finally, to make a quick translation of your data from .dm3 file format to the numpy, you may use the this [notebook](https://colab.research.google.com/github/pycroscopy/AICrystallographer/blob/master/AtomNet/dm3-to-numpy-v2.ipynb).
