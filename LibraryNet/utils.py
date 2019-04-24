#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maxim Ziatdinov
"""

import h5py
import json
import numpy as np
import cv2
from collections import OrderedDict

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

def open_library_hdf(filepath, *args):
    '''Opens an hdf5 file with experimental image and defect coordinates'''
    try:
        atoms = args[0]
    except IndexError:
        atoms = None
    with h5py.File(filepath, 'r') as f:
        image_data = f['nn_input'][:]
        scan_size = f['nn_input'].attrs['scan size']
        coordinates_all = np.empty((0, 3))
        for k in f.keys():
            if k != 'nn_input':
                coordinates = f[k].value
                coordinates = np.array(coordinates, dtype = 'U32')
                coordinates_all = np.append(coordinates_all, coordinates, axis=0)
    if atoms is not None:
        atomlist = coordinates_all[:, -1]
        xy = np.array(coordinates_all[:, :2], dtype=np.float)
        xy[:,[0, 1]] = xy[:,[1, 0]]
        atomlist[atomlist==atoms['lattice_atom']] = 0
        atomlist[atomlist==atoms['dopant']] = 1
        atomlist = np.array(atomlist, dtype=np.float)
        coordinates_all = np.concatenate((xy, atomlist[:, None]), axis=1)
        sort_idx = np.argsort(coordinates_all[:,-1])
        coordinates_all = coordinates_all[sort_idx]
    return image_data, scan_size, coordinates_all

def optimize_image_size(image_data, scan_size, px2ang=0.128, divisible_by=8):
    '''Adjusts the size of input image for getting
       the optimal decoding result with a neural network'''
    if np.amax(image_data) > 255:
        image_data = image_data/np.amax(image_data)
    image_size = image_data.shape[0]
    px2ang_i = image_data.shape[0]/scan_size
    # get optimal image dimensions for nn-based decoding
    image_size_new = np.around(image_size * (px2ang/px2ang_i))
    while image_size_new % divisible_by != 0:
        px2ang_i -= 0.001
        image_size_new = np.around(image_size * (px2ang/px2ang_i))
    # resize image if necessary
    image_data = cv2.resize(image_data, (int(image_size_new), int(image_size_new)), cv2.INTER_CUBIC)
    print('Image resized to {} by {}'.format(int(image_size_new),int(image_size_new)))
    return image_data

def atom_bond_dict(atom1='C', atom2='Si',
                   bond11=('C', 'C', 190),
                   bond12=('C', 'Si', 210),
                   bond22=('Si', 'Si', 250)):
    '''Returns type of host lattice atom, type of impurity atom
       and maximum bond lengths between each pair'''
    atoms = OrderedDict()
    atoms['lattice_atom'] = atom1
    atoms['dopant'] = atom2
    approx_max_bonds = {(bond11[0], bond11[1]): bond11[2],
                        (bond12[0], bond12[1]): bond12[2],
                        (bond22[0], bond22[1]): bond12[2]}
    return atoms, approx_max_bonds

def strainfunction(molecule_coord1, molecule_coord2, nnd_max=2):
    '''@Author: Xin Li CNMS/ORNL'''
    ##############
    # given points_ref and points_tar
    # estimator strain
    # Input: points_ref, points_target
    # Return:
    # F_est
    # t_est: translation vector
    # E_est: strain tensor
    # R_est: rotation matrix
    #############
    if len(molecule_coord1) != len(molecule_coord2):
        print('The defect structure is likely broken due to large strain',\
              'or you need to check a search radius')
        return
    points_ref = molecule_coord1.T
    points_tar = molecule_coord2.T
    n_points = points_tar.shape[1]
    y = points_tar.T.reshape(1,2*n_points).T
    M = np.zeros((2*n_points,6))
    for i in range(n_points):
        M[2*i:2*(i+1),:]=np.array(
            [[1,0,points_ref[0,i],points_ref[1,i],0,0],[0,1,0,0,points_ref[0,i],points_ref[1,i]]])
    x = np.linalg.pinv(M)@y
    t_est = x[0:2]
    F_est = x[2:].reshape(2,2)
    w,v = LA.eig(F_est.T@F_est)
    E_est = v.T@np.array([[np.sqrt(w[0]),0],[0,np.sqrt(w[1])]])@v
    R_est = F_est@np.linalg.inv(E_est)
    out = OrderedDict()
    out['translational vector'] = t_est
    out['strain_tensor'] = E_est
    out['rotation_matrix'] = R_est
    return out
