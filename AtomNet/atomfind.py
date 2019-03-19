# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 19:29:57 2019

@author: Maxim Ziatdinov

"""

import torch
import time
import numpy as np
import cv2
from scipy import ndimage


class dl_image:
    '''
    Image decoder with a trained neural network
    '''
    def __init__(self, image_data, model, *args, nb_classes=1,
                 max_pool_n=3, norm=1, use_gpu=False,
                 histogram_equalization=False):
        '''
        Args:
            image_data (ndarray): image stack or a single image (all greyscale)
            model: trained pytorch model
            nb_classes: number of classes in the model
            max_pool_n: number of max-pooling layers in the model
            norm: image normalization to 1
            use_gpu: optional use of gpu device for inference
            histogram_equalization: Equilazes image histogram
            args: tuple with image width and heigh for resizing operation
        '''
        if image_data.ndim == 2:
            image_data = np.expand_dims(image_data, axis=0)
        self.image_data = image_data
        try:
            self.rs = args[0]
        except IndexError:
            self.rs = image_data.shape[1:3]
        self.model = model
        self.nb_classes = nb_classes
        self.max_pool_n = max_pool_n
        self.norm = norm
        self.use_gpu = use_gpu
        self.hist_equ = histogram_equalization
    
    def img_resize(self):
        '''Image resizing (optional)'''
        if self.image_data.shape[1:3] == self.rs:
            return self.image_data.copy()
        image_data_r = np.zeros((self.image_data.shape[0],
                                 self.rs[0], self.rs[1]))
        for i, img in enumerate(self.image_data):
            img = cv2.resize(img, (self.rs[0], self.rs[1]))
            image_data_r[i, :, :] = img
        return image_data_r
        
    def img_pad(self, *args):
        '''Pads the image if its size (w, h)
        is not divisible by 2**n, where n is a number
        of max-pooling layers in a network'''
        try:
            image_data_p = args[0]
        except IndexError:
            image_data_p = self.image_data
        # Pad image rows (height)
        while image_data_p.shape[1] % 2**self.max_pool_n != 0:
            d0, _, d2 = image_data_p.shape
            image_data_p = np.concatenate(
                (image_data_p, np.zeros((d0, 1, d2))), axis=1)
        # Pad image columns (width)
        while image_data_p.shape[2] % 2**self.max_pool_n != 0:
            d0, d1, _ = image_data_p.shape
            image_data_p = np.concatenate(
                (image_data_p, np.zeros((d0, d1, 1))), axis=2)

        return image_data_p

    def hist_equalize(self, *args, number_bins=5):
        '''Histogram equalization (optional)'''
        try:
            image_data_ = args[0]
        except IndexError:
            image_data_ = self.image_data
     
        def equalize(image):
            image_hist, bins = np.histogram(image.flatten(), number_bins)
            cdf = image_hist.cumsum()
            #cdf_normalized = cdf * image_hist.max()/ cdf.max()
            image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
            return image_equalized.reshape(image.shape)

        image_data_h = np.zeros(shape=image_data_.shape)
        for i, img in enumerate(image_data_):
            img = equalize(img)
            image_data_h[i, :, :] = img

        return image_data_h


    def torch_format(self, image_data_):
        '''Reshapes and normalizes (optionally) image data
        to make it compatible with pytorch format'''
        image_data_ = np.expand_dims(image_data_, axis=1)
        if self.norm != 0:
            image_data_ = (image_data_ - np.amin(image_data_))/np.ptp(image_data_)
        image_data_ = torch.from_numpy(image_data_).float()
        return image_data_
    
    def predict(self, images):
        '''Returns probability of each pixel
           in image belonging to an atom'''
        if self.use_gpu:
            self.model.cuda()
            images = images.cuda()
        self.model.eval()
        with torch.no_grad():
            prob = self.model.forward(images)
            if self.nb_classes > 1:
                prob = torch.exp(prob)
        if self.use_gpu:
            self.model.cpu()
            images = images.cpu()
            prob = prob.cpu()
        prob = prob.permute(0, 2, 3, 1) # reshape with channel=last as in tf/keras
        prob = prob.numpy()

        return prob
    
    def decode(self):
        '''Make prediction'''
        image_data_ = self.img_resize()
        if self.hist_equ:
            image_data_ = self.hist_equalize(image_data_)
        image_data_ = self.img_pad(image_data_)
        image_data_torch = self.torch_format(image_data_)
        start_time = time.time()
        decoded_imgs = self.predict(image_data_torch)
        n_images_str = " image was " if decoded_imgs.shape[0] == 1 else " images were "
        print(str(decoded_imgs.shape[0]) + n_images_str + "decoded in approximately "
              + str(np.around(time.time() - start_time, decimals=2)) + ' seconds')
        images_data_torch = image_data_torch.permute(0, 2, 3, 1)
        images_numpy = images_data_torch.numpy()
        return images_numpy, decoded_imgs


class find_atoms:
    '''
    Transforms pixel data from decoded images
    into  a structure 'file' of atoms coordinates
    '''
    def __init__(self, decoded_imgs, threshold = 0.5, verbose = 1):
        '''
        Args:
            decoded_imgs: the output of a neural network (softmax/sigmoid layer)
            threshold: value at which the neural network output is thresholded
        '''
        if decoded_imgs.shape[-1] == 1:
            decoded_imgs_b = 1 - decoded_imgs
            decoded_imgs = np.concatenate((decoded_imgs[:, :, :, None],
                                           decoded_imgs_b[:, :, :, None]),
                                           axis=3)
        self.decoded_imgs = decoded_imgs
        self.threshold = threshold
        self.verbose = verbose
                       
    def get_all_coordinates(self, dist_edge=5):
        '''Extract all atomic coordinates in image
        via CoM method & store data as a dictionary
        (key: frame number)'''
        def find_com(image_data):
            '''Find atoms via center of mass methods'''
            labels, nlabels = ndimage.label(image_data)
            coordinates = np.array(
                ndimage.center_of_mass(image_data, labels,
                                       np.arange(nlabels)+1))
            coordinates = coordinates.reshape(coordinates.shape[0], 2)
            return coordinates

        d_coord = {}
        for i, decoded_img in enumerate(self.decoded_imgs):
            coordinates = np.empty((0, 2))
            category = np.empty((0, 1))
            # we assume that class backgrpund is always the last one
            for ch in range(decoded_img.shape[2]-1):
                _, decoded_img_c = cv2.threshold(decoded_img[:, :, ch],
                                                 self.threshold, 1, cv2.THRESH_BINARY)
                coord = find_com(decoded_img_c)
                coord_ch = self.rem_edge_coord(coord, dist_edge)
                category_ch = np.zeros((coord_ch.shape[0], 1))+ch
                coordinates = np.append(coordinates, coord_ch, axis=0)
                category = np.append(category, category_ch, axis=0)
            d_coord[i] = np.concatenate((coordinates, category), axis = 1)
        if self.verbose == 1:
            print("Atomic/defect coordinates extracted")
        return d_coord

    def rem_edge_coord(self, coordinates, dist_edge):
        '''Remove coordinates at the image edges; can be applied
           to coordinates without image as well (use Image = None
           when initializing "find_atoms" class)'''
        def coord_edges(coordinates, w, h, dist_edge):
            return [coordinates[0] > w - dist_edge,
                    coordinates[0] < dist_edge,
                    coordinates[1] > h - dist_edge,
                    coordinates[1] < dist_edge]
        if self.decoded_imgs is not None:
            if self.decoded_imgs.ndim == 3:
                w, h = self.decoded_imgs.shape[0:2]
            else:
                w, h = self.decoded_imgs.shape[1:3]
        else:
            w = np.amax(coordinates[:, 0] - np.amin(coordinates[:, 0]))
            h = np.amax(coordinates[:, 1] - np.amin(coordinates[:, 1]))
        coord_to_rem = [idx for idx, c in enumerate(coordinates) if any(coord_edges(c, w, h, dist_edge))]
        coord_to_rem = np.array(coord_to_rem, dtype = int)
        coordinates = np.delete(coordinates, coord_to_rem, axis=0)
        return coordinates
