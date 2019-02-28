import torch

try:
    device = torch.cuda.get_device_name(0)
    print('Available GPU device:', str(device))
except RuntimeError:
    print('If you would like to train a neural network,', \
    'please select GPU as a hardware accelerator', \
    'in Runtime --> Change runtime type',\
    'or in Edit --> Notebook settings')
