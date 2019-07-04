FROM python:3.7

RUN  python3 -m pip install numpy scipy matplotlib scikit-image scikit-learn opencv-python h5py pandas jupyterlab gdown \
     https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl \
     tensorflow==2.0.0-beta1  
