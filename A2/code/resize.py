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


def getGradImage(image):
    """
    Calculate the gradient image.

    Input: 
    image: the image after being applied the Gaussian blur.
    
    Output: the gradient image.
    """
    rows, cols = image.shape
    # from handout
    sobelx = np.array([[-1,  0, 1],
              [-2,  0,  2],
              [-1,  0,  1]])

    sobely = np.array([[-1, -2, -1],
              [0 ,  0,  0],
              [1 ,  2,  1]])
    # calculate the gradient matrix in x and y direction respectively.
    gradx = cv2.filter2D(image, -1, sobelx)
    grady = cv2.filter2D(image, -1, sobely)

    return np.sqrt(gradx**2 + grady**2)


@numba.jit(nopython=True)
def smallestPath(gradImage):
    """
    Find the shortest path on the gradient image.

    Input: 
    gradImage: a gradient image

    Output: 
    A 2D array f where f is calculated by dynamic programming and f[row][col]
    represents the smallest sum of gradients on the connected path that starts from 
    a pixel at row row and column col and goes down row by row until it finishes in 
    the last row of the image.
    """
    rows, cols = gradImage.shape
    f = np.empty((rows, cols))
    # init the dp array.
    f[rows-1] = gradImage[rows-1]
    for row in range(rows-2, -1, -1):
        for col in range(cols):
            # Bellman equation
            f[row, col] = f[row+1, col]
            if col-1 >= 0:
                f[row, col] = min(f[row, col], f[row+1, col-1])
            if col+1 < cols:
                f[row, col] = min(f[row, col], f[row+1, col+1])
            f[row, col] += gradImage[row, col]
    return f


@numba.jit(nopython=True)
def getDeletePixels(gradImage):
    """
    Calculate which pixels are needed to be deleted.

    Input:
    gradImage: A gradient image that need to be resize.

    Output:
    A 1D array deletePixels, where deletePixels[row] represents the 
    column number of pixel need to be deleted.
    """
    f = smallestPath(gradImage)
    rows, cols = gradImage.shape
    # Find start point of the shortest path at the beginning.
    start = 0
    currMin = f[0, 0]
    for col in range(cols):
        if f[0, col] < currMin:
            start = col
            currMin = f[0, col]
    deletePixels = [start]
    last = start
    # Find the shortest path.
    for row in range(1, rows):
        if last+1 == cols: # right most column
            if f[row, last] <= f[row, last-1]: deletePixels.append(last)
            else: deletePixels.append(last-1)
        elif last-1 == -1: # left most column
            if f[row, last] <= f[row, last+1]: deletePixels.append(last)
            else: deletePixels.append(last+1)
        else: # middle
            if f[row, last] <= f[row, last+1] and f[row, last] <= f[row, last-1]: 
                deletePixels.append(last)
            elif f[row, last-1] <= f[row, last] and f[row, last-1] <= f[row, last+1]:
                deletePixels.append(last-1)
            else:
                deletePixels.append(last+1)
        last = deletePixels[-1]
    return deletePixels


def deleteColumn(image, gradImg):
    """
    Delete a column with the smallest sum on gradImage on the image.

    Input:
    image: the rgb image that we want to resize
    gradImg: the gradient image of the rgb image

    Output:
    An RGB image that with one column less. (column deleted image)
    """
    rows, cols = gradImg.shape
    # find pixels needed to be deleted.
    deletePixels = getDeletePixels(gradImg)

    # delete the pixels of the images.
    mask = np.ones_like(gradImg, dtype=np.bool)
    for row in range(rows):
        mask[row, deletePixels[row]] = False
    mask = np.stack([mask] * 3, axis=2)
    img = image[mask].reshape((rows, cols-1, 3))
    return img


def seamCarving(img, desiredSize):
    """
    Apply seam carving algorithm on the image.

    Input:
    img: An RGB image that needed to be resized.
    desiredSize: The size of the output image. (row and column)

    Output:
    An RGB image with desiredSize. (row and column)
    """
    rows, cols, _ = img.shape
    erows, ecols = desiredSize
    output = img
    # delete columns
    while cols > ecols:
        grayImg = rgb2gray(output)
        gradImg = getGradImage(grayImg)
        output = deleteColumn(output, gradImg)
        rows, cols, _ = output.shape
    print("finish column carving.")
    # delete rows by rotate the image first.
    output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
    rows, cols, _ = output.shape
    while cols > erows:
        grayImg = rgb2gray(output)
        gradImg = getGradImage(grayImg)
        output = deleteColumn(output, gradImg)
        rows, cols, _ = output.shape
    print('finish row carving')
    # after row deletion rotate it back.
    output = cv2.rotate(output, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return output


def showResult(path, desiredSize):
    """Show result with the image path and desiredSize."""
    image = mpimg.imread(path)
    plt.figure(figsize=(16, 16))
    plt.subplot(2, 2, 1)
    plt.title('Original')
    plt.imshow(image)
    
    plt.subplot(2, 2, 2)
    plt.title('Seam Carving')
    seamCarvingResult = seamCarving(image, desiredSize)
    plt.imshow(seamCarvingResult)
    
    plt.subplot(2, 2, 3)
    plt.title('Crop')
    plt.imshow(image[:desiredSize[0], :desiredSize[1], :])
    
    plt.subplot(2, 2, 4)
    plt.title('Scale')
    plt.imshow(cv2.resize(image, (desiredSize[1], desiredSize[0])))
    
    plt.show()


def main():
    """main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('desiredSize', type=int, metavar='N', nargs='+', help='desired size of the image')
    args = parser.parse_args()
    print('Start seam carving...')
    showResult(args.path, args.desiredSize)


if __name__ == '__main__':
    main()