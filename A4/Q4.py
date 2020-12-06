import numpy as np
import numpy.linalg as LA

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

import plotly.graph_objects as go
from mpldatacursor import datacursor
import mplcursors

import os
import sys

import PIL


def calculate_homography(pts1, pts2):
    """
    Input points pairs
    Return the homography matrix.
    """
    matrix = []
    assert pts1.shape == pts2.shape
    n, _ = pts1.shape
    for i in range(n):
        x = pts1[i, 0]
        y = pts1[i, 1]
        xp = pts2[i, 0]
        yp = pts2[i, 1]
        matrix.extend([x,y,1,0,0,0,-1*xp*x,-1*xp*y,-1*xp])
        matrix.extend([0,0,0,x,y,1,-1*yp*x,-1*yp*y,-1*yp])
    matrix = np.array(matrix).reshape((2*n, 9))
    w, v = LA.eig(matrix.T @ matrix)
    smallest_index = np.argmin(w)
    h = v[:, smallest_index]
    return h.reshape((3,3))


def rgb2gray(rgb):
    """
    Input an rgb image
    Return a gray image
    """
    return 0.2989*rgb[:,:,0]+0.5870*rgb[:,:,1]+0.1140*rgb[:,:,2]


if __name__ == "__main__":
    image1 = mpimg.imread('A4Q4/hallway1.jpg')
    image2 = mpimg.imread('A4Q4/hallway2.jpg')
    image3 = mpimg.imread('A4Q4/hallway3.jpg')
    
    if len(sys.argv) == 1:
        print("Please give the command line argument (A/B/C)")
    else:    
        question = sys.argv[1]
        if question.lower().strip() == 'a':
            PIL.Image.open("Q4(1)A.png").show()

            apts1 = np.array([[998.7, 75.6],[1062.3, 12.9],[1094.7,228.2],[789.6, 614.5],[837.8,660.1]])
            apts2 = np.array([[845.1, 391.4],[900.5,331.2],[946.6, 537.],[666.4,937.5],[717.,981.5]])
            print("Calculated homography matrix is: ")
            print(calculate_homography(apts1, apts2))
            
            plt.imshow(rgb2gray(image1), cmap='gray')
            for i in range(5):
                plt.plot(apts1[i, 0], apts1[i, 1], "rs")
            plt.show()

            homography = calculate_homography(apts1, apts2)
            plt.imshow(rgb2gray(image2), cmap='gray')
            for i in range(5):
                plt.plot(apts2[i, 0], apts2[i, 1], "rs")
                x, y = apts1[i, 0], apts1[i, 1]
                pt = np.array([x, y, 1])
                predict = homography @ pt
                ax = predict[0]
                ay = predict[1]
                a = predict[2]
                plt.plot(ax/a, ay/a, "gs")
            plt.show()
        elif question.lower().strip() == 'b':
            PIL.Image.open("Q4(1)B.png").show()

            bpts1 = np.array([[999., 75.1],[1061.2, 11.6],[947.7,392.1],[788.3, 613.5],[1161.7,792.7]])
            bpts2 = np.array([[901.5, 265.9],[935.8,204.3],[882.3,581.7],[795.3,806.9],[1022.2,985.8]])
            print("Calculated homography matrix is: ")
            print(calculate_homography(bpts1, bpts2))

            plt.imshow(rgb2gray(image1), cmap='gray')
            for i in range(5):
                plt.plot(bpts1[i, 0], bpts1[i, 1], "rs")
            plt.show()

            homography = calculate_homography(bpts1, bpts2)
            plt.imshow(rgb2gray(image2), cmap='gray')
            for i in range(5):
                plt.plot(bpts2[i, 0], bpts2[i, 1], "rs")
                x, y = bpts1[i, 0], bpts1[i, 1]
                pt = np.array([x, y, 1])
                predict = homography @ pt
                ax = predict[0]
                ay = predict[1]
                a = predict[2]
                plt.plot(ax/a, ay/a, "gs")
            plt.show()
        elif question.lower().strip() == 'c':
            PIL.Image.open("Q4(1)C.png").show()

            cpts1 = np.array([[652.8,544.3],[789.4,613.8],[840.3,660.3],[482.5,777.3],[536.3,645.2]])
            cpts2 = np.array([[687.4,739.9],[795.1,807.4],[826.4,854.3],[414.9,982.1],[528.2,845.3]])
            calculate_homography(cpts1, cpts2)

            plt.imshow(rgb2gray(image1), cmap='gray')
            for i in range(5):
                plt.plot(cpts1[i, 0], cpts1[i, 1], "rs")
            plt.show()

            homography = calculate_homography(cpts1, cpts2)
            plt.imshow(rgb2gray(image3), cmap='gray')
            for i in range(5):
                plt.plot(cpts2[i, 0], cpts2[i, 1], "rs")
                x, y = cpts1[i, 0], cpts1[i, 1]
                pt = np.array([x, y, 1])
                predict = homography @ pt
                ax = predict[0]
                ay = predict[1]
                a = predict[2]
                plt.plot(ax/a, ay/a, "gs")
            plt.show()
        else:
            print("Please make sure the command line argument is A/B/C")