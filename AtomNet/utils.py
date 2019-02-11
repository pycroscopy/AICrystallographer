#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 00:30:02 2019

@author: ziatdinovmax
"""

import h5py
import json

def open_hdf(filepath):
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
