Status: <b>active project</b><br>
<p align="justify">
Please feel free to suggest possible modifications of a workflow to better suite the microscopy/materials community needs.</p>

# AtomNet
<p align="justify">
Application of a fully convolutional neural network for locating atoms in noisy experimental (scanning) transmission electron microscopy ((S)TEM) data. Based on our paper in ACS Nano, 2017, 11 (12), pp 12742–12752, but now with a better model (gives "cleaner" predictions) and using PyTorch instead of Keras. Please make sure to go through AtomFinderTutorial.ipynb for details regarding a typical model's usage and its limitation. We have uploaded models for graphene-like lattice structures and cubic lattices (<110> and <100> projections) and we expect to upload more models for different systems (and update the current one as well) in the near future.<br><br>
<p align="center">
  <img src="https://github.com/pycroscopy/AICrystallographer/blob/master/AtomNet/DL.png" width="75%" title="DL">
<p align="justify">
<br><br>
Currently it accepts inputs in the form of 2D and 3D numpy arrays (single images and stack of images). We plan to incorporate Pycroscopy translators in the near future to make it posssible working directly with .dm3 and other popular file formats.
  
## How to use
Once you go through ['AtomFinderTutorial' notebook](https://colab.research.google.com/github/pycroscopy/AICrystallographer/blob/master/AtomNet/AtomFinderTutorial.ipynb), use ['AtomFinder' notebook](https://colab.research.google.com/github/ziatdinovmax/AICrystallographer/blob/master/AtomNet/AtomFinder.ipynb) to analyze your data (follow the instructions in that notebook). Notice that once you click on 'Open in Colab' icon in a GitHub-opened notebook (or use ['Open in Colab'](https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo?hl=en) browser extension), the opened notebook will be completely independent from the notebook in this repository, so feel free to make any changes. You may want, however, to save it to your google drive to make sure that the changes won't be lost. Currently, one can use AtomNet for (S)TEM images from graphene-like systems and cubic lattices (<110> and <100> projections). We will be uploading more models in the near future. Let us know what specific type of systems you would like to investigate and we will try to prioritize them.<br><br>

<p align="center">
<img src="https://drive.google.com/uc?export=view&id=1akRH_mBR0dT_ejtCNURQmjP4lzRngK39" width="75%" title="AICrystallographer_AtomFinder">
