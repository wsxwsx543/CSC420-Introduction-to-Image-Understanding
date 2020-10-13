import sys
import argparse
import queue

import math
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

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


def visualize(size, sigma):
    """
    Visualize the Guassian Kernel in 3D shape.

    Input: 
    size: A tuple (H, W) represents the size of the Gaussian Kernel. 
          where H is the height of the matrix and w is the width of the matrix.
    sigma: the standard deviation of the kernel.
    """

    # get the Gaussian kernel with the inputs
    kernel = getGaussianKernel(size[0], size[1], sigma)

    # visualization
    fig = plt.figure()
    # Reference Document: https://matplotlib.org/3.1.1/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html
    ax = Axes3D(fig)
    X = np.arange(-(size[0]-1)/2, (size[0]-1)/2+1, 1)
    Y = np.arange(-(size[1]-1)/2, (size[1]-1)/2+1, 1)
    X, Y = np.meshgrid(X, Y)
    Z = kernel
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.title('sigma = {}, with kernel size = {}'.format(sigma, size))
    plt.show()


def getGaussianBlur(image, ksize=5, sigma=1):
    """
    Apply Guassian Blur to the input image. (i.e. use Gaussion to convolve with
    the input image)
    
    Input: 
    image: the input image with one channel.
    ksize: the squared kernel size (default value 5)
    sigma: the standard deviation of Gaussian kernel.

    Output:
    An one channel image after convolving with Gaussian Kernel.
    """

    rows, cols = image.shape
    padding = ksize - 1
    # get the gaussian kernel
    kernel = getGaussianKernel(ksize, ksize, sigma)
    # add padding to the image
    imgTmp = np.zeros([rows+2*padding, cols+2*padding])
    imgTmp[padding:rows+padding, padding:cols+padding] = image[:, :]
    # initialize  the output to all zeros
    output = np.zeros([rows+2*(ksize//2), cols+2*(ksize//2)])
    # convolve the Gaussian kernel with the image. Since the kernel is
    # centrosymmetric, I can use correlation directly.
    for i in range(rows+2*(ksize//2)):
        for j in range(cols+2*(ksize//2)):
            output[i][j] = np.sum(kernel * imgTmp[i:i+ksize, j:j+ksize])
    return output


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
    
    def sobelConv(image, sobel):
        """
        Helper function for getGradImage.

        Input:
        image: the image that used to calculate the gradient.
        sobel: sobelx or sobely, use sobel kernel to convolve with the iamge.

        Output:
        the image after being applied to convolution with sobel kernel.
        """
        # get padding image at the beginning
        imgTmp = np.zeros([rows+4, cols+4])
        imgTmp[2:rows+2, 2:cols+2] = image[:, :]
        # init the output image with all zeros
        output = np.zeros([rows+2, cols+2])
        # calculation convolution.
        for i in range(0, rows+2):
            for j in range(0, cols+2):
                tmpi = i + 1
                tmpj = j + 1
                for u in range(-1, 2):
                    for v in range(-1, 2):
                        output[i][j] += sobel[1+u][1+v] * imgTmp[tmpi-u][tmpj-v]
        return output
    
    # calculate the gradient matrix in x and y direction respectively.
    gradx = sobelConv(image, sobelx)
    grady = sobelConv(image, sobely)
    
    return np.sqrt(gradx**2 + grady**2)


def edgeDetection(gradImage):
    """
    Calculate the final edge image(edges are with pixel value 255 and other with 
    pixel value 0) with the auto-threshold algorithm (In Question 4).

    Input:
    gradImage: the gradient image after applying the getGradImage.

    Output:
    A one-channel edge image. 
    """
    rows, cols = gradImage.shape
    currTau = np.sum(gradImage) / (rows * cols)
    prevTau = -1
    edges = []
    output = np.zeros((rows, cols))
    # applying the auto-threshold algorithm
    while prevTau == -1 or abs(currTau - prevTau) > 1e-5:
        output = np.zeros((rows, cols))
        Lval = []
        Hval = []
        # binary the image
        for i in range(rows):
            for j in range(cols):
                if gradImage[i][j] < currTau:
                    Lval.append(gradImage[i][j])
                    output[i][j] = 0
                else:
                    Hval.append(gradImage[i][j])
                    output[i][j] = 255
        # update the threshold
        prevTau = currTau
        currTau = (sum(Hval)/len(Hval) + sum(Lval)/len(Lval)) / 2
    
    return output


def rgb2gray(rgb):
    """
    Transfer the rgb image to gray image.
    """
    # from discussion board
    return np.dot(rgb[..., :3], [0.2125, 0.7154, 0.0721])


def getResult(path):
    """
    Given a path and use edge detection algorithm from the beginning and 
    show the progress and output.
    """
    image = mpimg.imread(path)
    
    # get gray image
    grayImage = rgb2gray(image)
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.title('Gray Image')
    plt.imshow(grayImage, cmap='gray')
    plt.show()
    
    print("Get gray image.")
    blurredImage = getGaussianBlur(grayImage, sigma=2)
    print("Get blurred image.")
    gradImage = getGradImage(blurredImage)
    shownGradImage = gradImage / np.max(gradImage) * 255.0
    print("Get gradient image.")
    edgesImage = edgeDetection(gradImage)
    print("Get final edge image.")

    plt.figure(figsize=(15,5))
    plt.subplot(1, 3, 1)
    plt.title('Blurred Image')
    plt.imshow(blurredImage, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('Gradient Image')
    plt.imshow(shownGradImage, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('Edges Image')
    plt.imshow(edgesImage, cmap='gray')
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('imagePath', type=str)
    parser.add_argument('--visualize_kernel', type=bool, default=False, choices=[True, False], help='visualize the kernel whether or not.')
    args = parser.parse_args()

    if args.visualize_kernel:
        visualize((31, 31), 5)
        visualize((31, 31), 2)
    
    getResult(args.imagePath)