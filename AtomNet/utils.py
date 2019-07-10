#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 00:30:02 2019

@author: Maxim Ziatdinov
"""

import torch
import h5py
import json


class Hook():
    """
    Returns the input and output of a
    layer during forward/backward pass

    see https://www.kaggle.com/sironghuang/
        understanding-pytorch-hooks/notebook
    """
    def __init__(self, module, backward=False):
        """
        Args:
            module: torch modul(single layer or sequential block)
            backward (bool): replace forward_hook with backward_hook
        """
        if backward is False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

def mock_forward(model, dims=(1, 64, 64)):
    '''Passes a dummy variable throuh a network'''
    x = torch.randn(1, dims[0], dims[1], dims[2])
    out = model(x)
    return out

def open_hdf(filepath):
    '''Opens a custom hdf5 file with STEM image and reads the key metadata'''
    with h5py.File(filepath, 'r') as f:
        image_data = f['image_data'][:]
        metadata = json.loads(f['metadata'][()])
        if image_data.ndim == 2:
            n_images = 1
            w, h = image_data.shape
            print_s = 'image of the size'
        else:
            n_images = image_data.shape[0]
            w, h = image_data.shape[1:3]
            print_s = 'images of the size'
        print("Loaded", n_images, print_s, w, 'by', h)
        print("Sample name:", metadata["sample name"])
        print("Type of experiment:", metadata["type of data"])
    return image_data, metadata
