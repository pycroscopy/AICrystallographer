# -*- coding: utf-8 -*-
"""
Augmentation of image-label pairs (noise, zoom, rotation, resize)

@author: Maxim Ziatdinov

"""

import numpy as np
from scipy import ndimage
import cv2
from sklearn.utils import shuffle
from skimage import exposure
from skimage.util import random_noise
import random


class cropper:
    """
    Augments an input image-mask pair by performing image cropping procedure
    """
    def __init__(self, image, mask, window_size, step_size, batch_size):
        """
        Args:
            image (2d ndarray): image to be cropped (height x width),
            mask (3d ndarray): mask/ground truth (height x width x channels),
            window_size (tuple): width and height of sliding window,
            step_size (float): step size of sliding window,
            batch_size (int): number of images to return.
        """
        self.image = image
        if np.ndim(mask) == 2:
            mask = np.expand_dims(mask, axis=2)
        self.mask = mask
        self.window_size = window_size
        self.step_size = step_size
        self.batch_size = batch_size

    def sliding_window(self):
        """
        Returns the portion of the input image lying within sliding window
        """
        for y in range(0, self.image.shape[0], self.step_size):
            for x in range(0, self.image.shape[1], self.step_size):
                yield (self.image[y:y + self.window_size[1], x:x + self.window_size[0]],
                       self.mask[y:y + self.window_size[1], x:x + self.window_size[0], :])

    def imgen(self):
        """
        Returns a batch of cropped images and
        a batch of corresponding labels (ground truth)
        """
        X_batch = np.empty((0, self.window_size[0], self.window_size[1]))
        y_batch = np.empty((0, self.window_size[0], self.window_size[1], self.mask.shape[-1]))
        for window in self.sliding_window():
            if window[0].shape != self.window_size:
                continue
            X_batch = np.append(X_batch, [window[0]], axis=0)
            y_batch = np.append(y_batch, [window[1]], axis=0)
        X_batch, y_batch = shuffle(X_batch, y_batch)
        X_batch = X_batch[0:self.batch_size]
        y_batch = y_batch[0:self.batch_size]
        return X_batch, y_batch


class data_transform:
    """
    Applies a sequence of pre-defined operations for data augmentation.
    """
    def __init__(self, batch_size, width, height,
                 channels, dim_order='pytorch',
                 norm=1, **kwargs):
        """
        Args:
            batch_size (int): number of images in the batch,
            width (int): width of images in the batch,
            height (int): height of images in the batch,
            channels (int): number of classes (channels) in the ground truth
            dim_order (str): channel first (pytorch) or channel last (otherwise) ordering
            norm (int): normalization to 1,
            **flip (bool): image vertical/horizonal flipping,
            **rotate90 (bool): rotating image by +- 90 deg,
            **zoom (tuple): values for zooming-in (min height, max height, step);
              assumes height==width,
            **noise (dict): dictionary of noise values for each type of noise,
            **resize (tuple): values for image resizing (min height, max height, step);
              assumes heght==width.
        """
        self.n, self.w, self.h = batch_size, width, height
        self.ch = channels
        self.dim_order = dim_order
        self.norm = norm
        self.flip = kwargs.get('flip')
        self.rotate90 = kwargs.get('rotate90')
        self.zoom = kwargs.get('zoom')
        self.noise = kwargs.get('noise')
        self.resize = kwargs.get('resize')

    def transform(self, images, masks):
        """
        Applies a sequence of augmentation procedures
        to images and (except for noise) ground truth
        """
        images = (images - np.amin(images))/np.ptp(images)
        if self.flip:
            images, masks = self.batch_flip(images, masks)
        if self.noise is not None:
            images, masks = self.batch_noise(images, masks)
        if self.zoom is not None:
            images, masks = self.batch_zoom(images, masks)
        if self.resize is not None:
            images, masks = self.batch_resize(images, masks)
        if self.dim_order == 'pytorch':
            images = np.expand_dims(images, axis=1)
            masks = np.transpose(masks, (0, 3, 1, 2))
        else:
            images = np.expand_dims(images, axis=3)
            images = images.astype('float32')
        if self.norm != 0:
            images = (images - np.amin(images))/np.ptp(images)
        return images, masks

    def batch_noise(self, X_batch, y_batch,):
        """
        Takes an image stack and applies
        various types of noise to each image
        """
        def make_pnoise(image, l):
            vals = len(np.unique(image))
            vals = (l/50) ** np.ceil(np.log2(vals))
            image_n_filt = np.random.poisson(image * vals) / float(vals)
            return image_n_filt
        pnoise_range = self.noise['poisson']
        spnoise_range = self.noise['salt and pepper']
        gnoise_range = self.noise['gauss']
        blevel_range = self.noise['blur']
        c_level_range = self.noise['contrast']
        X_batch_a = np.zeros((self.n, self.w, self.h))
        for i, img in enumerate(X_batch):
            pnoise = random.randint(pnoise_range[0], pnoise_range[1])
            spnoise = random.randint(spnoise_range[0], spnoise_range[1])
            gnoise = random.randint(gnoise_range[0], gnoise_range[1])
            blevel = random.randint(blevel_range[0], blevel_range[1])
            clevel = random.randint(c_level_range[0], c_level_range[1])
            img = ndimage.filters.gaussian_filter(img, blevel*1e-1)
            img = make_pnoise(img, pnoise)
            img = random_noise(img, mode='gaussian', var=gnoise*1e-4)
            img = random_noise(img, mode='pepper', amount=spnoise*1e-3)
            img = random_noise(img, mode='salt', amount=spnoise*5e-4)
            img = exposure.adjust_gamma(img, clevel*1e-1)
            X_batch_a[i, :, :] = img
        return X_batch_a, y_batch

    def batch_zoom(self, X_batch, y_batch):
        """
        Crops and then resizes to the original size
        all images in one batch
        """
        zoom_list = np.arange(self.zoom[0], self.zoom[1], self.zoom[2])
        X_batch_a = np.zeros((self.n, self.w, self.h))
        y_batch_a = np.zeros((self.n, self.w, self.h, self.ch))
        for i, (img, gt) in enumerate(zip(X_batch, y_batch)):
            rs = np.random.choice(zoom_list)
            w1 = int((self.w-rs)/2)
            w2 = int(rs + (self.w-rs)/2)
            h1 = int((self.h-rs)/2)
            h2 = int(rs + (self.h-rs)/2)
            img = img[w1:w2, h1:h2]
            gt = gt[w1:w2, h1:h2]
            img = cv2.resize(img, (self.w, self.h))
            gt = cv2.resize(gt, (self.w, self.h))
            _, gt = cv2.threshold(gt, 0.25, 1, cv2.THRESH_BINARY)
            if len(gt.shape) != 3:
                gt = np.expand_dims(gt, axis=2)
            X_batch_a[i, :, :] = img
            y_batch_a[i, :, :, :] = gt
        return X_batch_a, y_batch_a

    def batch_flip(self, X_batch, y_batch):
        """
        Flips and rotates all images and in one batch
        and correponding labels (ground truth)
        """
        X_batch_a = np.zeros((self.n, self.w, self.h))
        y_batch_a = np.zeros((self.n, self.w, self.h, self.ch))
        int_r = (-1, 3) if self.rotate90 else (-1, 1)
        for i, (img, gt) in enumerate(zip(X_batch, y_batch)):
            flip_type = random.randint(int_r[0], int_r[1])
            if flip_type == 3:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
            elif flip_type == 2:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                img = cv2.flip(img, flip_type)
                gt = cv2.flip(gt, flip_type)
            if len(gt.shape) != 3:
                gt = np.expand_dims(gt, axis=2)
            X_batch_a[i, :, :] = img
            y_batch_a[i, :, :, :] = gt
        return X_batch_a, y_batch_a

    def batch_resize(self, X_batch, y_batch):
        """
        Resize all images in one batch and
        corresponding labels (ground truth)
        """
        rs_arr = np.arange(self.resize[0], self.resize[1], self.resize[2])
        rs = np.random.choice(rs_arr)
        if X_batch.shape[1:3] == (rs, rs):
            return X_batch, y_batch
        X_batch_a = np.zeros((self.n, rs, rs))
        y_batch_a = np.zeros((self.n, rs, rs, self.ch))
        for i, (img, gt) in enumerate(zip(X_batch, y_batch)):
            img = cv2.resize(img, (rs, rs), cv2.INTER_CUBIC)
            gt = cv2.resize(gt, (rs, rs), cv2.INTER_CUBIC)
            _, gt = cv2.threshold(gt, 0.25, 1, cv2.THRESH_BINARY)
            if len(gt.shape) < 3:
                gt = np.expand_dims(gt, axis=-1)
            X_batch_a[i, :, :] = img
            y_batch_a[i, :, :, :] = gt
        return X_batch_a, y_batch_a
