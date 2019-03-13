#!/bin/bash
echo "\nCopying files from DefectNet directory..."
cp -r AICrystallographer/DefectNet/* . 
echo "\nDownloading experimental data...\n"
gdown https://drive.google.com/uc?id=1Zz9iSr5eUsV9wHUzebbdp_Szr6GUiibf
mv 'WS2stack_subset.npy' exp_data/
echo "\nDownloading and installing PyTorch..."
pip install https://download.pytorch.org/whl/cu90/torch-1.0.0-cp36-cp36m-linux_x86_64.whl > /dev/null
echo "Checking if PyTorch sees Colab GPU device..."
python AICrystallographer/FerroNet/colabdevice.py
echo "\nInstalling packages for vizualization of neural networks..."
pip install pydot && apt-get install graphviz > /dev/null
pip install git+https://github.com/szagoruyko/pytorchviz > /dev/null
