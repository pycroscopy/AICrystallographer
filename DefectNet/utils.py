"""
util functions

@author: Maxim Ziatdinov
"""

import torch
import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def torch_format(images, norm=1):
    '''Reshapes dimensions, normalizes (optionally)
       and converts image data to a pytorch float tensor.
       (assumes mage data is stored as numpy array)'''
    if images.ndim == 2:
        images = np.expand_dims(images, axis=0)
    images = np.expand_dims(images, axis=1)
    if norm != 0:
        images = (images-np.amin(images))/np.ptp(images)
    images = torch.from_numpy(images).float()
    return images


def predict(images, model, gpu=False):
    '''Returns probability (as seen by neural network)
    of each pixel in image belonging to a defect'''
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


def threshold_output(imgdata, t=0.5):
    '''Binary threshold of an output of a neural network'''
    imgdata_ = cv2.threshold(imgdata, t, 1, cv2.THRESH_BINARY)[1]
    return imgdata_


def filter_isolated_cells(imgdata, th=150):
    '''Filters out blobs above cetrain size
    in the thresholded neural network output'''
    label_img, cc_num = ndimage.label(imgdata)
    cc_areas = ndimage.sum(imgdata, label_img, range(cc_num + 1))
    area_mask = (cc_areas > th)
    label_img[area_mask[label_img]] = 0
    label_img[label_img > 0] = 1
    return label_img


def find_blobs(imgdata):
    '''Finds position of defects in the processed output
       of a neural network via center of mass method'''
    labels, nlabels = ndimage.label(imgdata)
    coordinates =  ndimage.center_of_mass(
        imgdata, labels, np.arange(nlabels)+1)
    coordinates = np.array(coordinates, dtype=np.float)
    coordinates = coordinates.reshape(coordinates.shape[0], 2)
    return coordinates


def remove_edge_coord(imgdata, coordinates, dist_edge=20):
    '''Removes coordinates too close to image edges'''
    def coord_edges(coordinates, w, h, dist_edge):
        return [coordinates[0] > w - dist_edge,
                coordinates[0] < dist_edge,
                coordinates[1] > h - dist_edge,
                coordinates[1] < dist_edge]
    w, h = imgdata.shape[0:2]
    coord_to_rem = [idx for idx, c in enumerate(coordinates) if
                    any(coord_edges(c, w, h, dist_edge))]
    coord_to_rem = np.array(coord_to_rem, dtype=int)
    coordinates = np.delete(coordinates, coord_to_rem, axis=0)
    return coordinates


def draw_boxes(imgdata, defcoord, bbox=16, figsize_=(6, 6)):
    '''Draws boxes cetered around the extracted dedects'''
    fig, ax = plt.subplots(1, 1, figsize=figsize_)
    ax.imshow(imgdata, cmap='gray')
    for point in defcoord:
        startx = int(round(point[0] - bbox))
        starty = int(round(point[1] - bbox))
        p = patches.Rectangle(
            (starty, startx), bbox*2, bbox*2,
            fill=False, edgecolor='red', lw=2)
        ax.add_patch(p)
    ax.grid(False)
    plt.show()


def inference(imgdata, model, thresh=0.3, thresh_blobs=160):
    '''Obtain position of defects from the neural network output'''
    decoded_imgs = predict(torch_format(imgdata), model=model)
    decoded_imgs = threshold_output(decoded_imgs.numpy()[0,:,:,0], t=thresh)
    decoded_imgs = filter_isolated_cells(decoded_imgs, th=thresh_blobs)
    defcoord = find_blobs(decoded_imgs)
    defcoord = remove_edge_coord(imgdata, defcoord, 20 )
    return defcoord
