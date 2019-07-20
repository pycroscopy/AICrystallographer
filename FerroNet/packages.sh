#!/bin/bash

checkgpu() {
python - <<END
import torch
device_count = torch.cuda.device_count()
if device_count > 0:
    print('Available GPU devices:')
    for i in range(device_count):
        print(torch.cuda.get_device_name(i))
else:
    print('No GPU device detected.',\
    'If using Colab and would like to train a neural network,', \
    'please select GPU as a hardware accelerator', \
    'in Runtime --> Change runtime type',\
    'or in Edit --> Notebook settings')
END
}

if test "$1" = "modules"; then
    echo "Checking if PyTorch sees Colab GPU device..."
    checkgpu
    echo "\n\nInstalling packages for vizualization of neural networks -->\n"
    pip install pydot && apt-get install graphviz
    pip install git+https://github.com/szagoruyko/pytorchviz
elif test "$1" = "dataset"; then
    echo "Downloading training set -->\n"
    gdown https://drive.google.com/uc?id=10wo3fWl6lbD0p7S_gJ241BKJ05v6KL2U
else 
    echo "No match found. Use <module> for installation of additional python libraries or <dataset> to download  a training set"
fi
