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

echo "\nCopying files from DefectNet directory..."
cp -r AICrystallographer/DefectNet/* . 
echo "\nDownloading experimental data...\n"
gdown https://drive.google.com/uc?id=1Zz9iSr5eUsV9wHUzebbdp_Szr6GUiibf
mv 'WS2stack4_subset.npy' exp_data/
python -c "import torch" 2>/dev/null
if test $? -ne 0; then
    echo "\nDownloading and installing PyTorch -->\n"
    pip install https://download.pytorch.org/whl/cu90/torch-1.0.0-cp36-cp36m-linux_x86_64.whl    
else
    echo "\nFound existing installation of PyTorch"
    pip show torch | grep Version
fi
echo "\nChecking if PyTorch sees Colab GPU device..."
colabdevice
echo "\nInstalling packages for vizualization of neural networks..."
pip install pydot && apt-get install graphviz > /dev/null
pip install git+https://github.com/szagoruyko/pytorchviz > /dev/null
