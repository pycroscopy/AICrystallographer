#!/bin/bash
if test "$1" = "modules"; then
    echo "\n\nCopying files from FeroNet directory..."
    cp -r AICrystallographer/FerroNet/* . 
    echo "Completed"
    echo "\n\nDownloading and installing PyTorch -->\n"
    pip install https://download.pytorch.org/whl/cu90/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
    echo "\n\nChecking if PyTorch sees Colab GPU device..."
    python AICrystallographer/FerroNet/colabdevice.py
    echo "\n\nInstalling packages for vizualization of neural networks -->\n"
    pip install pydot && apt-get install graphviz
    pip install git+https://github.com/szagoruyko/pytorchviz
elif test "$1" = "dataset"; then
    echo "Downloading training set -->\n"
    gdown https://drive.google.com/uc?id=10wo3fWl6lbD0p7S_gJ241BKJ05v6KL2U
else 
    echo "No match found. Use <module> for installation of additional python libraries or <dataset> to download  a training set"
fi
