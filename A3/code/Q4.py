import numpy as np
from numpy import linalg as LA

import os
import math

from scipy import ndimage, misc
import random

import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2gray

def calc_gradient_orientation(gray, threshold=-1):
    """Calculate the magnitude and the orientation of the gradient image."""
    # find grad on x, y respectively
    gradx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grady = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # avoid division by zero error
    gradx[gradx==0] = 1e-6
    # calculate magnitude
    magnitude = np.sqrt(gradx*gradx + grady*grady)
    magnitude[magnitude<threshold] = 0
    # calculate orientation
    orientation = np.arctan(grady/gradx)
    orientation = np.where(orientation<0, orientation+np.pi, orientation)
    return magnitude, orientation

def calc_hog(mag, ori):
    """Calculate the histogram of gradient based on the magnitude and orientation."""
    assert mag.shape == ori.shape
    H = mag.shape[0] // 8
    W = mag.shape[1] // 8
    result = np.zeros((H, W, 6))
    # calculate the HoG
    for h in range(H):
        for w in range(W):
            row = h * 8 + (mag.shape[0]-H*8)//2
            col = w * 8 + (mag.shape[1]-W*8)//2
            for r in range(row, row+8):
                for c in range(col, col+8):
                    result[h, w, int((ori[r, c]/(np.pi/6))+0.5)%6] += mag[r, c]
    return result

def visualize(img, hog, tau):
    """Visualize the result"""
    if len(img.shape) == 2:
        h, w = img.shape
    else:
        h, w, _ = img.shape
    m = h // tau
    n = w // tau
    
    # plot quiver plot.
    theta = np.linspace(0, 2*np.pi, 16)
    r = np.linspace(0, 1, 6)
    x = np.sin(theta)[:,np.newaxis]*r
    y = np.cos(theta)[:,np.newaxis]*r
    
    X, Y = np.meshgrid(np.linspace(tau/2, (n-1)*tau+tau/2, n),
                       np.linspace(tau/2, (m-1)*tau+tau/2, m))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap='gray')
    for i in range(6):
        ax.quiver(X, Y,
                  np.sin(i*np.pi/6) * hog[:, :, i]*0.001,
                  np.cos(i*np.pi/6) * hog[:, :, i]*0.001,
                  color='red',
                  linewidth=0.5,
                  headlength=0,
                  headwidth=1,
                  headaxislength=0,
                  pivot='middle')
    plt.show()

def build_descriptors(hog):
    """calculate the descriptor. Return (M-1)x(N-1)x24 array."""
    H, W, _ = hog.shape
    result = np.zeros((H-1, W-1, 24))
    for h in range(H-1):
        for w in range(W-1):
            result[h, w, :] = np.concatenate([hog[h, w, :], hog[h+1, w, :], hog[h, w+1, :], hog[h+1, w+1]])
            # normalize
            result[h, w, :] = result[h, w, :] / np.sqrt(np.sum(result[h, w, :] * result[h, w, :])+1e-6)
    return result

def show_result(path, threshold=-1):
    image=mpimg.imread(path)
    gray = rgb2gray(image)
    mag, ori = calc_gradient_orientation(gray, threshold)
    hog = calc_hog(mag, ori)
    descriptor = build_descriptors(hog)
    visualize(image, hog, 8)
    return descriptor
  

descriptor = show_result('Q4/Q4/1.jpg')
np.savetxt('1.txt', descriptor.reshape((1, -1)), delimiter=',')
descriptor = show_result('Q4/Q4/2.jpg')
np.savetxt('2.txt', descriptor.reshape((1, -1)), delimiter=',')
descriptor = show_result('Q4/Q4/3.jpg')
np.savetxt('3.txt', descriptor.reshape((1, -1)), delimiter=',')
