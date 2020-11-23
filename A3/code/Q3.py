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

def read_images(path):
    """Read images from the give path, Return gray scale images."""
    files = os.listdir(path)
    images = []
    for file in files:
        image = mpimg.imread(path + '/' + file)
        image = rgb2gray(image)
        images.append(image)
    return np.array(images)


def gaussian_blur(images, grad_sigma=5, time_sigma=1):
    """Apply Gaussian blur on those images and apply gaussian blur on time axis."""
    N, H, W = images.shape
    result = []
    # Gaussian blur on the image
    for n in range(N):
        blurred = ndimage.gaussian_filter(images[n,:,:], grad_sigma)
        result.append(blurred)
    result = np.array(result, dtype=np.float32)
    # Gaussian blur on time axis
    result = ndimage.gaussian_filter1d(result, time_sigma, axis=0)
    return result


def lucus_kanade_iter_refinement(image1, image2, block_size, eps, lmbda_thre=1e-3, lmbda_ratio_thre=100, uv_thre=0.1):
    """Calculate u, v according to the lucus kanade iterative refinement algorithm."""
    H, W = image1.shape
    resultu = np.empty((H//block_size, W//block_size), dtype=np.float32)
    resultv = np.empty((H//block_size, W//block_size), dtype=np.float32)
    # calculate the gradient matrix.
    gradx1 = cv2.Sobel(image1, cv2.CV_64F, 1, 0, ksize=3)
    grady1 = cv2.Sobel(image1, cv2.CV_64F, 0, 1, ksize=3)
    gradx2 = cv2.Sobel(image2, cv2.CV_64F, 1, 0, ksize=3)
    grady2 = cv2.Sobel(image2, cv2.CV_64F, 0, 1, ksize=3)
    for h in range(H//block_size):
        for w in range(W//block_size):
            # some init
            x, y = w*block_size, h*block_size
            curru, currv = float('inf'), float('inf')
            u, v = 0.0, 0.0
            f = False
            cnt = 0
            # while the change is not small
            while math.sqrt(curru**2+currv**2) >= eps:
                cnt += 1
                xp, yp = int(x+u), int(y+v)
                # select a block
                Ix = gradx1[y:y+block_size, x:x+block_size]
                Iy = grady1[y:y+block_size, x:x+block_size]
                # calculate Ix2, Iy2, IxIy
                Ix2 = np.sum(Ix*Ix)
                Iy2 = np.sum(Iy*Iy)
                IxIy = np.sum(Ix*Iy)
                # calculate updated matrix and the It
                moved = image2[yp:yp+block_size, xp:xp+block_size]
                moved_h, moved_w = moved.shape
                It = moved-image1[y:y+moved_h, x:x+moved_w]
                # Calculate the second moment matrix
                IxIt = np.sum(Ix[:moved_h, :moved_w]*It)
                IyIt = np.sum(Iy[:moved_h, :moved_w]*It)
                left = np.array([[Ix2, IxIy],[IxIy, Iy2]], dtype=np.float32)
                # eigenvalue decomposition
                lmbdas, _ = LA.eig(left)
                lmbda1 = lmbdas[0]
                lmbda2 = lmbdas[1]
                # some condition for valid lambda
                if lmbda1 == 0 or lmbda2 == 0 or min(lmbda1,lmbda2)<lmbda_thre or max(lmbda1, lmbda2)/min(lmbda1, lmbda2)>lmbda_ratio_thre:
                    resultu[h, w] = 0
                    resultv[h, w] = 0
                    f = True
                    break
                # solve the matrix
                uv = np.dot(LA.inv(left), -1*np.array([IxIt, IyIt]))
                curru, currv = uv[0], uv[1]
                # update u, v
                u, v = u + curru, v + currv
                if cnt >= 10: break
            if f: continue
            if math.sqrt(u**2 + v**2) < uv_thre:
                resultu[h, w] = 0
                resultv[h, w] = 0
            else:
                resultu[h, w] = u
                resultv[h, w] = v
    return resultu, resultv


def lucus_kanade_one_shot(image1, image2, block_size, lmbda_thre=1e-3, lmbda_ratio_thre=100, uv_thre=0.1):
    """Rerturn the u, v vectors on the image."""
    H, W = image1.shape
    resultu = np.empty((H//block_size, W//block_size), dtype=np.float32)
    resultv = np.empty((H//block_size, W//block_size), dtype=np.float32)
    # calculate the gradient matrix
    gradx1 = cv2.Sobel(image1, cv2.CV_64F, 1, 0, ksize=3)
    grady1 = cv2.Sobel(image1, cv2.CV_64F, 0, 1, ksize=3)
    gradx2 = cv2.Sobel(image2, cv2.CV_64F, 1, 0, ksize=3)
    grady2 = cv2.Sobel(image2, cv2.CV_64F, 0, 1, ksize=3)
    for h in range(H//block_size):
        for w in range(W//block_size):
            # fix a block
            x, y = w*block_size, h*block_size
            # find Ix, Iy in the block
            Ix = gradx1[y:y+block_size, x:x+block_size]
            Iy = grady1[y:y+block_size, x:x+block_size]
            # calculate Ix2, Iy2, IxIy
            Ix2 = np.sum(Ix*Ix)
            Iy2 = np.sum(Iy*Iy)
            IxIy = np.sum(Ix*Iy)
            # Calculate It
            It = image2[y:y+block_size, x:x+block_size]-image1[y:y+block_size, x:x+block_size]
            # Calculate IxIt, IyIt
            IxIt = np.sum(Ix*It)
            IyIt = np.sum(Iy*It)
            left = np.array([[Ix2, IxIy],[IxIy, Iy2]], dtype=np.float32)
            lmbdas, _ = LA.eig(left)
            lmbda1 = lmbdas[0]
            lmbda2 = lmbdas[1]
            # sigular matrix or some lambda is less than the threshould or the ratio is too large.
            if lmbda1 == 0 or lmbda2 == 0 or min(lmbda1,lmbda2)<lmbda_thre or max(lmbda1, lmbda2)/min(lmbda1, lmbda2)>lmbda_ratio_thre:
                resultu[h, w] = 0
                resultv[h, w] = 0
                continue
            # Solve the matrix
            uv = np.dot(LA.inv(left), -1*np.array([IxIt, IyIt]))
            # if u, v is too small, set to 0.
            if math.sqrt(uv[0]**2 + uv[1]**2) <= uv_thre:
                resultu[h, w] = 0
                resultv[h, w] = 0
            else:
                resultu[h, w] = uv[0]
                resultv[h, w] = uv[1]

    return resultu, resultv


def visualize(img, block_size, u, v):
    assert u.shape == v.shape
    h, w = u.shape
    x,y = np.meshgrid(np.linspace(block_size/2, w*block_size+block_size/2, w), 
                      np.linspace(block_size/2, h*block_size+block_size/2, h))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap='gray')
    q = ax.quiver(x,y,u,-v, color='red')
    plt.show()

images = read_images('Q3_optical_flow/Q3_optical_flow/Army')
blurred = gaussian_blur(images, 3, 1)
u, v = lucus_kanade_iter_refinement(blurred[0,:,:], blurred[1,:,:], 5, 1, lmbda_thre=1e-2, lmbda_ratio_thre=100, uv_thre=0.1)
visualize(images[0,:,:], 5, u, v)

images = read_images('Q3_optical_flow/Q3_optical_flow/Backyard')
blurred = gaussian_blur(images, 3, 1)
u, v = lucus_kanade_iter_refinement(blurred[0,:,:], blurred[1,:,:], 11, 2, lmbda_thre=1e-2, lmbda_ratio_thre=10, uv_thre=0.3)
visualize(images[0,:,:], 11, u, v)

images = read_images('Q3_optical_flow/Q3_optical_flow/Basketball')
blurred = gaussian_blur(images, 5, 1)
u, v = lucus_kanade_iter_refinement(blurred[6,:,:], blurred[7,:,:], 7, 1, lmbda_thre=3e-3, lmbda_ratio_thre=100, uv_thre=0.15)
visualize(images[6,:,:], 7, u, v)

images = read_images('Q3_optical_flow/Q3_optical_flow/Dumptruck')
blurred = gaussian_blur(images, 5, 1)
u, v = lucus_kanade_iter_refinement(blurred[5,:,:], blurred[6,:,:], 7, 3, lmbda_thre=1e-3, lmbda_ratio_thre=100, uv_thre=0.01)
visualize(images[5,:,:], 7, u, v)

images = read_images('Q3_optical_flow/Q3_optical_flow/Evergreen')
blurred = gaussian_blur(images, 5, 1)
u, v = lucus_kanade_iter_refinement(blurred[5,:,:], blurred[6,:,:], 11, 0.3, lmbda_thre=1e-4, lmbda_ratio_thre=200, uv_thre=1)
visualize(images[5,:,:], 11, u, v)

images = read_images('Q3_optical_flow/Q3_optical_flow/Grove')
blurred = gaussian_blur(images, 5, 1)
u, v = lucus_kanade_iter_refinement(blurred[5,:,:], blurred[6,:,:], 11, 1, lmbda_thre=1e-2, lmbda_ratio_thre=100, uv_thre=0.001)
visualize(images[5,:,:], 11, u, v)

images = read_images('Q3_optical_flow/Q3_optical_flow/Mequon')
blurred = gaussian_blur(images, 7, 1)
u, v = lucus_kanade_iter_refinement(blurred[0,:,:], blurred[1,:,:], 11, 2, lmbda_thre=5e-3, lmbda_ratio_thre=50, uv_thre=0.1)
visualize(images[0,:,:], 11, u, v)

images = read_images('Q3_optical_flow/Q3_optical_flow/Schefflera')
blurred = gaussian_blur(images, 7, 1)
u, v = lucus_kanade_iter_refinement(blurred[0,:,:], blurred[1,:,:], 11, 1, lmbda_thre=5e-3, lmbda_ratio_thre=50, uv_thre=0.1)
visualize(images[0,:,:], 11, u, v)

images = read_images('Q3_optical_flow/Q3_optical_flow/Urban')
blurred = gaussian_blur(images, 7, 3)
u, v = lucus_kanade_iter_refinement(blurred[0,:,:], blurred[1,:,:], 15, 1.5, lmbda_thre=5e-3, lmbda_ratio_thre=100, uv_thre=0.01)
visualize(images[0,:,:], 15, u, v)

images = read_images('Q3_optical_flow/Q3_optical_flow/Wooden')
blurred = gaussian_blur(images, 7, 3)
u, v = lucus_kanade_iter_refinement(blurred[0,:,:], blurred[1,:,:], 7, 1, lmbda_thre=5e-3, lmbda_ratio_thre=200, uv_thre=0.01)
visualize(images[0,:,:], 7, u, v)

images = read_images('Q3_optical_flow/Q3_optical_flow/Yosemite')
blurred = gaussian_blur(images, 5, 3)
u, v = lucus_kanade_iter_refinement(blurred[0,:,:], blurred[1,:,:], 5, 0.5, lmbda_thre=5e-3, lmbda_ratio_thre=200, uv_thre=0.001)
visualize(images[0,:,:], 5, u, v)
