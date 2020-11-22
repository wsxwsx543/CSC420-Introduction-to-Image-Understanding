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

def build_square(image_size, square_size, white_square=True):
    """
    Create a new square, if the white_square is True, then create a white square,
    otherwise create a black square.
    """
    if white_square:
        image = np.zeros((image_size, image_size))
        image[(image_size-square_size)//2:(image_size-square_size)//2+square_size, (image_size-square_size)//2:(image_size-square_size)//2+square_size] = 1
        return image
    else:
        image = np.ones((image_size, image_size))
        image[(image_size-square_size)//2:(image_size-square_size)//2+square_size, (image_size-square_size)//2:(image_size-square_size)//2+square_size] = 0
        return image


def show_result(length, start=0, end=0, num=1000):
    minresponse = []
    maxresponse = []
    sigma = np.linspace(start, end, num)
    for s in sigma:
        minresponse.append(np.min((s**2) * ndimage.gaussian_laplace(build_square(3*length, length, True), s, mode='constant')))
        maxresponse.append(np.max((s**2) * ndimage.gaussian_laplace(build_square(3*length, length, False), s, mode='constant')))
    
    # white square
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(sigma, minresponse)
    opt = sigma[np.argmin(minresponse)]
    plt.title('White Square: Side length={}, with optimal sigma={:.4}'.format(length, opt))
    plt.xlabel('sigma')
    plt.ylabel('response')
    
    # black square
    plt.subplot(1, 2,2)
    plt.plot(sigma, maxresponse)
    opt = sigma[np.argmax(maxresponse)]
    plt.title('Black Square: Side length={}, with optimal sigma={:.4}'.format(length, opt))
    plt.xlabel('sigma')
    plt.ylabel('response')
    
    plt.show()


if __name__ == "__main__":
    show_result(5, 1, 4)
    show_result(10, 2.5, 7)
    show_result(15, 4, 11)
    show_result(20, 4, 15)