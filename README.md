# AICrystallographer
To aid in the automated analysis of atomically resolved images, we will upload here our deep/machine learning models and 'workflows' (such as AtomNet, DefectNet, SymmetryNet, etc.) with the Jupyter notebooks describing in details how to perform the analysis. Most of the notebooks can be opened and executed in Google Colaboratory, which is a Jupyter notebook environment for machine learning research that requires no setup to use (and also provides free GPU/TPU).<br><br>
AI Crystallographer is an active project and we expect to be adding more workflows and pre-trained models in the near future. Currently it includes the following sub-packages:
<ul>
<li><b>DefectNet:</b> Complete workflow for locating atomic defects in electron microscopy movies with a convolutional neural network using only a single movie frame to generate a training set. It is based on our paper in npj Computational Materials 5, 12 (2019), but now with the updated augmentation procedure (includes adding noise, zoom-in and flip/rotations) and using PyTorch deep learning framework instead of the Keras one for model training/predictions.</li>
<li><b>AtomNet:</b> Application of a fully convolutional neural network for locating atoms in noisy experimental scanning transmission electron microscopy data from graphene. Based on our paper in ACS Nano 11, 12742 (2017), but now with a better model (gives "cleaner" predictions) and using PyTorch instead of Keras. The current model is limited to graphene lattice, but we expect to upload more models for different systems in the near future.</li>
<li><b>FerroNet:</b> Application of different machine learning and multivariate analysis tools (neural networks, dimensionality reduction, clustering/unmixing) for analysis of ferroic distortions in high-resolution scanning transmission electron microscopy data on perovskites. Based on our unpublished but to be submitted work.</li>
<li><b>SymmetryNet</b>: Application of a deep convolutional network used to determine 2D Bravais lattice symmetry from atomically resolved images. Based on our paper in npj Computational Materials 4, 30 (2018).</li>
<li><p align="justify"><b>Tutorials:</b> Tutorial-like notebooks on using class activation maps for locating defects in the images and using a fully convolutional neural network for cleaning atom-resolved data and locating atoms in it.
</ul>
<br>
<h2>How to use</h2>
  
The easiest way to use the tools in this package is by opening the notebooks in Google Colab either by clicking "Open in Colab" in the GitHub-opened notebook file or by installing a [Colab browser extension](https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo?hl=en). See our [tutorial notebook](https://colab.research.google.com/github/pycroscopy/AICrystallographer/blob/master/Tutorials/ColabNotebooks_BasicOperations.ipynb) for the best practices to use Google Colab notebooks.<br>
<p align="center">
<img src="https://drive.google.com/uc?export=view&id=1XCydssu0RIxlRKmKklfuP_wtqdXQV5fB" width="75%" title="DL">

You may also use this package via Docker container. 
First, clone the repository to your local machine. Then, from your terminal, ```cd``` into the cloned repository and run ```docker build -t aicr .``` command (you may substitute 'aicr' with whatever name you like). Once it finishes buiding a Docker image, run ```docker run -it -p 8080:8080 -v <path_to_directory_with_AICrystallographer>:/home aicr /bin/bash``` to start a container. You will now be able to launch a Jupyter notebook from inside your container by running ```jupyter lab --ip=0.0.0.0 --port=8080 --allow-root command``` command and then opening a http://localhost:8080 in your browser. 
