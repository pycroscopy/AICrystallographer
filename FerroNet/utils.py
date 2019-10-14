"""
Created on Tue Feb 12 14:20 2019

@author: ziatdinovmax
"""

import torch
import numpy as np
import h5py
import cv2
from scipy import ndimage
import scipy.spatial as spatial


class generate_batches:
    def __init__(self, hf_file, batch_size, *args, channel_order = 'tfkeras'):
        '''Creates a batch generator
        Args:
            hf_file: path to hdf5 file with training data
            batch_size: size of batch to be produced
            channel_order: reorders dimensions from channel_last to channel_first
            args (tuple): image resizing during training (min_size, max_size, step)
        '''
        self.f = h5py.File(hf_file, 'r')
        self.batch_size = batch_size
        self.channel_order = channel_order
        try:
            self.resize_ = args[0]
        except IndexError:
            self.resize_ = None
    
    def steps(self, mode='training'):
        """Estimates number of steps per epoch"""
        if mode == 'val':
            n_samples = self.f['X_test'][:].shape[0]
        else:
            n_samples = self.f['X_train'][:].shape[0]
        return np.arange(n_samples//self.batch_size)
    
    def batch(self, idx, mode='training'):
        """Generates batch of the selected size
        for training images and the corresponding
        ground truth"""
        def batch_resize(X_batch, y_batch, rs):
            '''Resize all images in one batch'''
            if X_batch.shape[1:3] == (rs, rs):
                return X_batch, y_batch
            X_batch_r = np.zeros((X_batch.shape[0], rs, rs, X_batch.shape[-1]))
            y_batch_r = np.zeros((y_batch.shape[0], rs, rs)) 
            for i, (img, gt) in enumerate(zip(X_batch, y_batch)):
                img = cv2.resize(img, (rs, rs))
                img = np.expand_dims(img, axis = 2)
                gt = cv2.resize(gt, (rs, rs))
                X_batch_r[i, :, :, :] = img
                y_batch_r[i, :, :] = gt
            return X_batch_r, y_batch_r
        if mode == 'val':
            X_batch = self.f['X_test'][int(idx*self.batch_size):int((idx+1)*self.batch_size)]
            y_batch = self.f['y_test'][int(idx*self.batch_size):int((idx+1)*self.batch_size)]
        else:
            X_batch = self.f['X_train'][int(idx*self.batch_size):int((idx+1)*self.batch_size)]
            y_batch = self.f['y_train'][int(idx*self.batch_size):int((idx+1)*self.batch_size)]
        if self.resize_ != None:
            rs_arr = np.arange(self.resize_[0], self.resize_[1], self.resize_[2])
            rs = np.random.choice(rs_arr)
            X_batch, y_batch = batch_resize(X_batch, y_batch, rs)
        X_batch = torch.from_numpy(X_batch).float()
        if self.channel_order == 'tfkeras':
            X_batch = X_batch.permute(0, 3, 1, 2)
        y_batch = torch.from_numpy(y_batch).long()
        yield X_batch, y_batch
            
    def close_(self):
        """Closes h5 file"""
        if self.f:
            self.f.close()
            self.f = None
            
def torch_format(images, norm=1, n_pooling=3):
    '''Reshapes dimensions, normalizes (optionally) 
       and converts image data to a pytorch float tensor.
       (assumes mage data is stored as numpy array)'''
    if images.ndim == 2:
        images = np.expand_dims(images, axis=0)
    images = np.expand_dims(images, axis=1)
    if norm != 0:
        images = (images - np.amin(images))/np.ptp(images)
    images = img_pad(images, n_pooling)
    #images = torch.from_numpy(images).float()
    return images

def predict(images, model, gpu=False):
    '''Returns probability of each pixel in image
        belonging to an atom of a particualr type'''
    if gpu:
        model.cuda()
        images = images.cuda()
    model.eval()
    with torch.no_grad():
        prob = model.forward(images)
    if gpu:
        model.cpu()
        images = images.cpu()
        prob = prob.cpu()
    prob = torch.exp(prob)
    prob = prob.permute(0, 2, 3, 1)
    return prob

def img_resize(image_data, rs):
    '''Image resizing'''
    image_data_r = np.zeros((image_data.shape[0], rs[0], rs[1]))
    for i, img in enumerate(image_data):
        img = cv2.resize(img, (rs[0], rs[1]))
        image_data_r[i, :, :] = img
    return image_data_r
    
def img_pad(image_data, n_pooling):
    '''Pads the image if its size (w, h)
    is not divisible by 2**n, where n is a number
    of max-pooling layers in a network'''
    # Pad image rows (height)
    image_data_p = np.copy(image_data)
    while image_data_p.shape[1] % 2**n_pooling != 0:
        d0, _, d2 = image_data_p.shape
        image_data_p = np.concatenate(
            (image_data_p, np.zeros((d0, 1, d2))), axis=1)
    # Pad image columns (width)
    while image_data_p.shape[2] % 2**n_pooling != 0:
        d0, d1, _ = image_data_p.shape
        image_data_p = np.concatenate(
            (image_data_p, np.zeros((d0, d1, 1))), axis=2)
    return image_data_p

def find_com(image_data, t=0.5):
    '''Returns center of the mass for all the blobs
       in each channel of network output'''
    thresh = cv2.threshold(image_data, t, 1, cv2.THRESH_BINARY)[1]
    lbls, nlbls = ndimage.label(thresh)
    com = np.array(ndimage.center_of_mass(
        thresh, lbls, np.arange(nlbls)+1))
    com = com.reshape(com.shape[0], 2)
    return com

def extract_subimages(network_output, com, d, nb_classes, **kwargs):
    '''Description TBA'''
    icut = kwargs.get('n')
    if icut == None:
        icut = len(com)
    img_cr_all = np.empty((0, int(d*2), int(d*2), nb_classes))
    com_ = np.empty((0, 2))
    for i, c in enumerate(com):
        cx = int(np.around(c[0]))
        cy = int(np.around(c[1]))
        img_cr = np.copy(
            network_output[0, cx-d:cx+d, cy-d:cy+d, 0:nb_classes])
        if img_cr.shape == (int(d*2), int(d*2), nb_classes):
            img_cr_all = np.append(img_cr_all, [img_cr], axis=0)
            com_ = np.append(com_, [c], axis=0)
        if i > icut:
            return img_cr_all, com_
    return img_cr_all, com_

def estimate_nnd(com1, com2, icut=500):
    '''Description TBA'''
    d0 = []
    for i, c in enumerate(com1):
        distance = spatial.cKDTree(com2).query(c)[0]
        d0.append(distance)
        if i > icut:
            break
    d0 = np.array(d0)
    d0 = np.mean(d0) + 0.5*np.std(d0)
    print('Average nearest-neighbor distance:', np.around(d0))
    return d0

def estimate_rad(input_image, t= 0.5, icut=500):
    '''Description TBA'''
    thresh = cv2.threshold(
    input_image, t, 1, cv2.THRESH_BINARY)[1]
    thresh = cv2.convertScaleAbs(thresh)
    contours = cv2.findContours(
        thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1]
    ma0 = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 1:
            _, (m, M), _ = cv2.fitEllipse(cnt)
            ma0.append(M)
        if i > icut:
            break
    ma0 = np.array(ma0)
    ma0 = 0.5*np.mean(ma0 + 0.5*np.std(ma0))
    print('Average blob radius:', np.around(ma0))
    return ma0

def color_list():
    '''Returns a list of colors for scatter/line plots'''
    return ['blue', 'green', 'red', 'magenta',
            'cyan', 'yellow', 'black', 'darkviolet',
            'darkorange', 'gold', 'pink']
