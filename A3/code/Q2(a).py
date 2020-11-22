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

def naive_ransac(src_pts, dst_pts, iters):
    """Return an affine matrix calculated by RANSAC."""
    assert len(src_pts) == len(dst_pts)
    length = len(src_pts)
    affine_matrix = None
    pretotal = -1
    for i in range(iters):
        # randomly pick three points.
        sample_index = random.sample(range(length), 3)
        Xs = src_pts[sample_index]
        Ys = dst_pts[sample_index]
        x0 = Xs[0, 0, :]
        x1 = Xs[1, 0, :]
        x2 = Xs[2, 0, :]
        y0 = Ys[0, 0, :]
        y1 = Ys[1, 0, :]
        y2 = Ys[2, 0, :]
        pts1 = np.array([x0, x1, x2])
        pts2 = np.array([y0, y1, y2])
        # calculate the model based on the randomly picked pts.
        M = cv2.getAffineTransform(pts1, pts2)
        inliers = 0
        # count number of inliers
        for i in range(length):
            dummy_src_pt = np.concatenate([src_pts[i, 0, :], np.array([1])])
            dst_hat = np.matmul(M, dummy_src_pt)
            dst_pt = dst_pts[i, 0, :]
            # L1 distance
            if np.sum(abs(dst_hat-dst_pt)) < 5:
                inliers += 1
        # update the model
        if inliers > pretotal:
            pretotal = inliers
            affine_matrix = M
    return affine_matrix


def show_comparison1(img1, img2, affine_matrix, src_pts, dst_pts):
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    img3 = img2.copy()
    img4 = img2.copy()
    
    # calculate homography matrix.
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    detected_book = cv2.perspectiveTransform(pts,M)
    img3 = cv2.polylines(img3,[np.int32(detected_book)],True,255,3, cv2.LINE_AA)
    
    # calculate RANSAC matrix result.
    detected_book = np.array([np.dot(affine_matrix, np.concatenate([pts[0,0,:], np.array([1])])), 
                              np.dot(affine_matrix, np.concatenate([pts[1,0,:], np.array([1])])), 
                              np.dot(affine_matrix, np.concatenate([pts[2,0,:], np.array([1])])), 
                              np.dot(affine_matrix, np.concatenate([pts[3,0,:], np.array([1])]))]).reshape(-1, 1, 2)
    img4 = cv2.polylines(img4,[np.int32(detected_book)],True,255,3, cv2.LINE_AA)
    print("Homography Matrix: ", M)
    print("Affine Matrix: ", affine_matrix)
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 2, 1)
    plt.title('Homography Matrix')
    plt.imshow(img3, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Affine Transformation with RANSAC')
    plt.imshow(img4, cmap='gray')
    plt.show()


if __name__ == "__main__":
    # load two images
    img1 = cv2.imread("Book_cover.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("Book_pic.png", cv2.IMREAD_GRAYSCALE)

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    # (brute force) matching of descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2) # k=2 means find the top two matchs for each query descriptor

    # Apply ratio test (as per David Lowe's SIFT paper: compare the best match with the 2nd best match_
    good_matches = []
    good_matches_without_list = []
    for m,n in matches:
        if m.distance < 0.75*n.distance: # only accept matchs that are considerably better than the 2nd best match
            good_matches.append([m])
            good_matches_without_list.append(m) # this is to simplify finding a homography later

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,
                            None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, 
                            matchColor=(0,255,0))
    # plt.imshow(img3),plt.show()

    # you can also an approximate (but fast) nearest neighbour algorithm called FLANN. See here:
    # https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches_without_list ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches_without_list ]).reshape(-1,1,2)

    # half of matches are inliers, use 40 iterations.
    M = naive_ransac(src_pts, dst_pts, 40)

    show_comparison1(img1, img2, M, src_pts, dst_pts)