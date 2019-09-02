"""
FFT subtraction module

@author: Artem Maksov, Kevin Stubbs

"""

import numpy as np
from scipy import fftpack
import cv2
import warnings
from numpy.linalg import norm
from sklearn import mixture


def FFTmask(imgsrc, maskratio=10):
    """
    Takes a square real space image and filter out a disk with radius equal to:
    1/maskratio * image size.
    Retruns FFT transform of the image and the filtered FFT transform
    """
    # Take the fourier transform of the image.
    F1 = fftpack.fft2((imgsrc))
    # Now shift so that low spatial frequencies are in the center.
    F2 = (fftpack.fftshift((F1)))
    # copy the array and zero out the center
    F3 = F2.copy()
    l = int(imgsrc.shape[0]/maskratio)
    m = int(imgsrc.shape[0]/2)
    y, x = np.ogrid[1: 2*l + 1, 1:2*l + 1]
    mask = (x - l)*(x - l) + (y - l)*(y - l) <= l*l
    F3[m-l:m+l, m-l:m+l] = F3[m-l:m+l, m-l:m+l] * (1 - mask)
    return F2, F3


def FFTsub(imgsrc, F3):
    """
    Takes real space image and filtred FFT.
    Reconstructs real space image and subtracts it from the original.
    Returns normalized image.
    """
    reconstruction = np.real(fftpack.ifft2(fftpack.ifftshift(F3)))
    diff = np.abs(imgsrc - reconstruction)
    # normalization
    diff = diff - np.amin(diff)
    diff = diff/np.amax(diff)
    return diff


def threshImg(diff, threshL=0.25, threshH=0.75):
    """
    Takes in difference image, low and high thresold values,
    and outputs a map of all defects.
    """
    threshIL = diff < threshL
    threshIH = diff > threshH
    threshI = threshIL + threshIH
    return threshI


def raaft(patch, which_norm, inner=0, outer=1):
    """
    Given an image patch and choice of norm, generate the RAAFT feature vector for that patch.

    Parameters
    ----------
    patch : 2D numpy array
        The patch which we want to apply the RAAFT to.
    which_norm : float
        Which norm to normalize the feature vector by. Can choose any real number
        for this value, but choosing 1 <= which_norm <= infty is the most natural.
        Choosing either 1 or 2 works well in most cases.
    inner, outer : float
        To help with noise it is sometimes useful to set parts of the Fourier transform
        of in each patch to zero. If a patch has dimensions (px, py) then the points 
        (kx,ky) in k-space which are not set to zero must satisfy: 
            inner * sqrt(px^2 + py^2) <= sqrt(kx^2 + ky^2) 
            outer * sqrt(px^2 + py^2) >= sqrt(kx^2 + ky^2)

    Returns
    -------
    feat_vector : 2D numpy array with shape (patch.shape[0], 1)
        The RAAFT feature vector corresponding to the given patch.
    """
    p_sz = patch.shape
    center = (p_sz[0]//2, p_sz[1]//2)
    rad = np.sqrt((p_sz[0]**2 + p_sz[1]**2) / 2)
    tol = 1e-10 # this so we don't accidentally divide by 0

    #
    #patch = (patch - np.mean(patch)) / np.std(patch)
    
    # Take the absolute value of the Fourier transform
    tmp_img = np.abs(fftpack.fftshift(cv2.dct(patch)))

    # Normalize the patch 
    tmp_img /= norm(np.ndarray.flatten(tmp_img), ord=which_norm) + tol

    # Perform the polar transform
    tmp_img = cv2.warpPolar(tmp_img, None, center, rad, cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)

    # Sum over the radial variable
    feat_vec = np.sum(tmp_img, 0)
    feat_vec[:int(rad*inner)] = 0
    feat_vec[int(rad*outer):] = 0
        
    return np.array(feat_vec, ndmin=2)


def raaftGenFeatureVectors(img, patch_sz, num_x, num_y, which_norm, inner=0, outer=1):
    """
    Given an image, a patch size, and the number of patches in x and y dimensions
    calculate all of the feature vectors for this image. A warning is raised if
    there are not enough patches to completely cover the image.
    
    Parameters
    ----------
    img : 2D numpy array
        The image to generate the feature vectors for.
    patch_sz : int
        The size of each patch.
    num_x, num_y : int
        The patches will be placed on an equally spaced grid with dimensions
        (num_x) by (num_y) 
    which_norm : float
        Which norm to normalize the feature vector by. Can choose any real number
        for this value, but choosing 1 <= which_norm <= infty is the most natural.
        Choosing either 1 or 2 works well in most cases.
    inner, outer : float
        To help with noise it is sometimes useful to set parts of the Fourier transform
        of in each patch to zero. If a patch has dimensions (px, py) then the points 
        (kx,ky) in k-space which are not set to zero must satisfy: 
            inner * sqrt(px^2 + py^2) <= sqrt(kx^2 + ky^2) 
            outer * sqrt(px^2 + py^2) >= sqrt(kx^2 + ky^2)

    Returns
    -------
    feat_vector_mat : 2D numpy array with shape (num_x * num_y, patch_sz)
        A matrix of feature vectors for each patch in the image
    """
    img_sz = img.shape

    too_small = patch_sz * num_y < img_sz[0] or \
                patch_sz * num_x < img_sz[1]
    if too_small:
        warnings.warn("Warning. Not enough patches to cover entire image.")

    x_step = (img_sz[0] - patch_sz) / (num_x - 1)
    y_step = (img_sz[1] - patch_sz) / (num_y - 1)

    # Calculate the size of the feature vector to preallocate an array
    x_rng = range(0, patch_sz)
    y_rng = range(0, patch_sz)
    fv_size = raaft(img[np.ix_(x_rng, y_rng)], which_norm, inner, outer).size
    feat_vecs_mat = np.zeros((num_x * num_y, fv_size))

    # Generate all of the feature vectors for the image
    for count, (idx_x, idx_y)  in enumerate(np.ndindex((num_x, num_y))):
        x_center = patch_sz // 2 + int(np.round(idx_x * x_step))
        y_center = patch_sz // 2 + int(np.round(idx_y * y_step))
        
        x_rng = range(x_center - patch_sz//2, x_center + patch_sz//2)
        y_rng = range(y_center - patch_sz//2, y_center + patch_sz//2)

        feat_vecs_mat[count,:] = raaft(img[np.ix_(x_rng, y_rng)], which_norm, inner, outer)

    return feat_vecs_mat
    

def raaftClusterFeatureVectors(feat_vecs, N):
    """
    Given an array of all the feature vectors, generate a Gaussian Mixture model with 
    the specified number of clusters

    Parameters
    ----------
    feat_vecs : 2D numpy array
        A matrix of feature vectors. 
    N : int
        The number of clusters to use for Gaussian Mixture model (GMM) clustering

    Returns
    -------
    clusters : 2D numpy array with shape (feat_vecs.shape[0], 1)
        The cluster assignment for each feature vector
    gmm : sklearn.mixture.BayesianGaussianMixture
        The learned GMM model
    """
    gmm = mixture.BayesianGaussianMixture(
        n_components=N,
        covariance_type='full',
        max_iter=5000,
        tol=1e-5).fit(feat_vecs)
    clusters = gmm.predict(feat_vecs)

    return clusters, gmm


def raaftGenResults(img_sz, clusters, N, patch_sz, num_x, num_y):
    """
    Given an image patch and choice of norm, generate the RAAFT feature vector for that patch.

    Parameters
    ----------
    img_sz : A 2-tuple
        The shape of the original image
    clusters : 2D numpy array
        The cluster assignment for each feature vector
    N : int
        The number of clusters used in Gaussian Mixture model clustering.
    patch_sz : int
        The size of each patch.
    num_x, num_y : int
        The the centers of patches will be placed on an equally spaced grid
        with dimensions (num_x) by (num_y).

    Returns
    -------
    mask_img : 3D numpy array with shape (N, img_sz[0], img_sz[1])
        A collection of masks (estimated probabilities) for each cluster assignment.
    """
    mask_img = np.zeros([N] + list(img_sz), dtype=float)
    count_img = np.zeros(img_sz, dtype=float)

    x_step = (img_sz[0] - patch_sz) / (num_x - 1)
    y_step = (img_sz[1] - patch_sz) / (num_y - 1)

    for count, (idx_x, idx_y)  in enumerate(np.ndindex((num_x, num_y))):
        x_center = patch_sz // 2 + int(np.round(idx_x * x_step))
        y_center = patch_sz // 2 + int(np.round(idx_y * y_step))
        
        x_rng = range(x_center - patch_sz//2, x_center + patch_sz//2)
        y_rng = range(y_center - patch_sz//2, y_center + patch_sz//2)

        clus_num = clusters[count]

        count_rng = np.ix_(x_rng, y_rng)
        mask_rng = np.ix_((clus_num,), x_rng, y_rng)

        count_img[count_rng] += 1
        mask_img[mask_rng] += 1

    for i in range(N):
        mask_img[i,:,:] /= count_img
        
    return mask_img


def raaftRun(p, img):
    """
    Runs Radially Averaged Absolute Fourier transform (RAAFT) method on the passed in image.

    Parameters
    ----------
    params : dict
        A dictionary with the (hyper)parameters for the RAAFT method. Required parameters are:
        "patch_size", "num_patches_x", "num_patches_y", "which_norm", and "num_clusters"
    img : 2D numpy array
        The image to run RAAFT analysis on

    Returns
    -------
    mask_img : 3D numpy array with shape (params["num_clusters"], img.shape[0], img.shape[1])
        A collection of masks (estimated probabilities) for each cluster assignment.
    """

    print("Generating feature vectors...", end = '', flush=True)
    feat_vecs = raaftGenFeatureVectors(img,
                                       p["patch_size"],
                                       p["num_patches_x"],
                                       p["num_patches_y"],
                                       p["which_norm"],
                                       p["inner"],
                                       p["outer"])
    print("done")

    print("Clustering...", end='', flush=True)
    clusters, _ = raaftClusterFeatureVectors(feat_vecs, p["num_clusters"])
    print("done")

    print("Generating masks...", end = '', flush=True)
    mask_img = raaftGenResults(img.shape,
                               clusters,
                               p["num_clusters"],
                               p["patch_size"],
                               p["num_patches_x"],
                               p["num_patches_y"])
    print("done")

    return mask_img
