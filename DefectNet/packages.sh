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
    'If using Google Colab and would like to train a neural network,', \
    'please select GPU as a hardware accelerator', \
    'in Runtime --> Change runtime type',\
    'or in Edit --> Notebook settings')
END
}

echo "\nDownloading experimental data...\n"
gdown https://drive.google.com/uc?id=1Zz9iSr5eUsV9wHUzebbdp_Szr6GUiibf
mv 'WS2stack4_subset.npy' exp_data/
echo "\nChecking if PyTorch sees GPU device..."
checkgpu
echo "\nInstalling packages for vizualization of neural networks..."
pip install pydot && apt-get install graphviz > /dev/null
pip install git+https://github.com/szagoruyko/pytorchviz > /dev/null
