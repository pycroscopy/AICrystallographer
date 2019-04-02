#!/bin/bash

colabdevice() {
python - <<END
import torch
device_count = torch.cuda.device_count()
if device_count > 0:
    print('Available GPU devices:')
    for i in range(device_count):
        print(torch.cuda.get_device_name(i))
else:
    print('No GPU device detected.',\
    'If you would like to train a neural network,', \
    'please select GPU as a hardware accelerator', \
    'in Runtime --> Change runtime type',\
    'or in Edit --> Notebook settings')
END
}

if test "$1" = "modules"; then
    echo "\n\nCopying files from FeroNet directory..."
    cp -r AICrystallographer/FerroNet/* . 
    echo "Completed"
    python -c "import torch" 2>/dev/null
    if test $? -ne 0; then
        echo "\n\nDownloading and installing PyTorch -->\n"
        pip install https://download.pytorch.org/whl/cu90/torch-1.0.0-cp36-cp36m-linux_x86_64.whl    
    else
        echo "Found existing installation of PyTorch"
        pip show torch | grep Version
    fi
    echo "\n\nChecking if PyTorch sees Colab GPU device..."
    colabdevice
    echo "\n\nInstalling packages for vizualization of neural networks -->\n"
    pip install pydot && apt-get install graphviz
    pip install git+https://github.com/szagoruyko/pytorchviz
elif test "$1" = "dataset"; then
    echo "Downloading training set -->\n"
    gdown https://drive.google.com/uc?id=10wo3fWl6lbD0p7S_gJ241BKJ05v6KL2U
else 
    echo "No match found. Use <module> for installation of additional python libraries or <dataset> to download  a training set"
fi
