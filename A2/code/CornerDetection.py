# import libraries
import sys
import argparse

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import math

import numpy as np
from numpy import linalg as LA

# I use numba to accelerate.
# Reference: https://numba.pydata.org/numba-doc/latest/user/jit.html
import numba
import cv2


def rgb2gray(rgb):
    """
    Transfer the rgb image to gray image.
    """
    # from discussion board
    return np.dot(rgb[..., :3], [0.2125, 0.7154, 0.0721])


# Reference (Gaussian distribution): https://en.wikipedia.org/wiki/Multivariate_normal_distribution
def getGaussianKernel(H=3, W=3, sigma=1):
    """
    Given a kernel size and a standard deviation. output the gaussian kernel.
    Input:
    H: height of the output kernel matrix (default value 3).
    W: width of the output kernel matrix (default value 3).
    sigma: standard deviation of the output matrix.

    Output:
    (H x W) numpy array, the Gaussian Kernel matrix with sd=sigma.
    """
    if sigma == 0:
        sigma = 1
    gaussMatrix = np.zeros([H, W], np.float32)

    # get the center
    cH = (H - 1) / 2
    cW = (W - 1) / 2
    
    # calculate gauss(sigma, r, c)
    for r in range(H):
        for c in range(W):
            norm2 = (r - cH)**2 + (c - cH)**2
            gaussMatrix[r][c] = math.exp(-norm2 / (2 * (sigma**2)))
    sumGM = np.sum(gaussMatrix)
    gaussKernel = gaussMatrix / sumGM
    return gaussKernel


def getGradXY(image):
    """
    Get the gradient image of the input image on X and Y axis respectively.

    Input:
    image: the input RGB image

    Output: A tuple
            First element represents the gradient image on X
            Second element represents the gradient image on Y
    """
    sobelx = np.array([[-1,  0, 1],
                      [-2,  0,  2],
                      [-1,  0,  1]])

    sobely = np.array([[-1, -2, -1],
                      [0 ,  0,  0],
                      [1 ,  2,  1]])
    gradx = cv2.filter2D(image, -1, sobelx)
    grady = cv2.filter2D(image, -1, sobely)
    assert gradx.shape == grady.shape
    return gradx, grady


def calcEigenvalue(gradx, grady, window):
    """
    Calculate eigenvalues on each pixel.

    Input:
    gradx: gradient image on X (Ix)
    grady: gradient image on Y (Iy)
    window: the window we need to use (w) (used to being Gaussian kernel)

    Output:
    A list lmbda, where 
    lmbda[0] represents the first eigenvalue of each pixel.
    lmbda[1] represents the second eigenvalue of each pixel. 
    """
    Ixy = gradx * grady
    Ix2 = gradx * gradx
    Iy2 = grady * grady
    
    # Since it is linear, I can calculate it in this way.
    Ixy = cv2.filter2D(Ixy, -1, window)
    Ix2 = cv2.filter2D(Ix2, -1, window)
    Iy2 = cv2.filter2D(Iy2, -1, window)
    
    lmbda = [[], []]
    
    # calculate the eigenvalues for each pixel.
    rows, cols = gradx.shape
    for row in range(rows):
        for col in range(cols):
            currMat = np.array([[Ix2[row, col], Ixy[row, col]],
                               [Ixy[row, col], Iy2[row, col]]])
            w, v = LA.eig(currMat)
            lmbda[0].append(w[0])
            lmbda[1].append(w[1])
    return lmbda


def cornerDetection(image, lmbda1, lmbda2, threshold):
    """
    Find the corners of the image.

    Input:
    image: the image where I find corners
    lmbda1: the first eigenvalue of each pixel of this image.
    lmbda2: the second eigenvalue of each pixel of this image.
    threshold: the threshold is set to distinguish whether a pixel is corner or not.

    Output:
    An RGB image with the corners are marked by red color.
    """
    rows, cols, _ = image.shape
    lmbda1 = np.array(lmbda1)
    lmbda2 = np.array(lmbda2)
    lmbda1 = lmbda1.reshape([rows, cols])
    lmbda2 = lmbda2.reshape([rows, cols])
    # find corners
    mask1 = lmbda1 > threshold
    mask2 = lmbda2 > threshold
    mask = np.bitwise_and(mask1, mask2)
    tmpImage = np.array(image)
    # mark the corners.
    tmpImage[mask] = [255, 0, 0]
    return tmpImage


def getResult(path, windowSize, windowSigma, threshold):
    """
    Give an image and window size, standard deviation and threshold to use to find the corners.
    """
    image = mpimg.imread(path)
    grayImage = rgb2gray(image)
    blurImage = cv2.GaussianBlur(grayImage, (3, 3), 1)
    gradx, grady = getGradXY(blurImage)
    
    window = getGaussianKernel(windowSize, windowSize, windowSigma)
    lmbda = calcEigenvalue(gradx, grady, window)
    
    plt.figure()
    plt.scatter(lmbda[0], lmbda[1])
    plt.show()
    
    resultImage = cornerDetection(image, lmbda[0], lmbda[1], threshold)
    plt.figure(figsize=(12,12))
    plt.imshow(resultImage)
    
    plt.show()


def main():
    """main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--windowSize', default=5, type=int, help='size of the window.')
    parser.add_argument('--windowSigma', default=1, type=float, help='standard deviation of the window.')
    parser.add_argument('--threshold', default=7000, type=int, help='threshold set to detect the corner.')

    args = parser.parse_args()

    getResult(args.path, args.windowSize, args.windowSigma, args.threshold)


if __name__ == '__main__':
    main()