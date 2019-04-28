Status: <b>active project</b><br>
<p align="justify">
Please feel free to suggest possible modifications of a workflow to better suite the microscopy/materials community needs.</p>

# AtomNet
<p align="justify">
Application of a fully convolutional neural network for locating atoms in noisy experimental scanning transmission electron microscopy data from graphene. Based on our paper in ACS Nano, 2017, 11 (12), pp 12742–12752, but now with a better model (gives "cleaner" predictions) and using PyTorch instead of Keras. Please make sure to go through GrapheneAtomFinder.ipynb for details regarding the model's usage and its limitation. The current model is limited to graphene lattice, but we expect to upload more models for different systems (and update the current one as well) in the near future.<br><br>
<p align="center">
  <img src="https://github.com/pycroscopy/AICrystallographer/blob/master/AtomNet/DL.png" width="75%" title="DL">
<p align="justify">
<br><br>
Currently it accepts inputs in the form of 2D and 3D numpy arrays (single images and stack of images). We plan to incorporate Pycroscopy translators in the near future to make it posssible working directly with .dm3 and other popular file formats.
