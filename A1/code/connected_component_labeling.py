import numpy as np

import math

import queue

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D


def autoThreshold(image):
    """
    Get the one-channel image and calculate the threshold by this algorithm then output the 
    binary image.

    Input:
    image: the image we need to binary

    Output:
    An one-channel binary image with all pixel values equal to 0 and 255 
    """
    rows, cols = image.shape
    currTau = np.sum(image) / (rows * cols)
    prevTau = -1
    edges = []
    output = np.zeros((rows, cols))
    while prevTau == -1 or abs(currTau - prevTau) > 1e-8:
        output = np.zeros((rows, cols))
        Lval = []
        Hval = []
        for i in range(rows):
            for j in range(cols):
                if image[i][j] < currTau:
                    Lval.append(image[i][j])
                    output[i][j] = 0
                else:
                    Hval.append(image[i][j])
                    output[i][j] = 255
        prevTau = currTau
        currTau = (sum(Hval)/len(Hval) + sum(Lval)/len(Lval)) / 2
    
    return output


def connected_component_labeling(image):
    """
    Given a binary image and labeling the connected-components in the image.

    Input:
    image: a binary image needed to be labeled.
    Output:
    A tuple: labelled_image, num_components
    labelled_image: the image after labelling, each components with a unique color.
    num_components: number of connected-components in this image.
    """

    # define some constants at the beginning.
    BACKGROUND = 0
    FOREGROUND = 255
    
    rows, cols = image.shape
    currLabel = 1
    labelledImage = np.zeros_like(image)
    for i in range(rows):
        for j in range(cols):
            # background or has been labelled before
            if image[i][j] == BACKGROUND or labelledImage[i][j] != 0:
                continue
            
            # current pixel is foreground pixel
            q = queue.Queue()
            labelledImage[i][j] = currLabel
            q.put( (i, j) )
            while not q.empty():
                x, y = q.get()
                
                # consider the surrounding 8 pixels
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        # if it equals to itself then don't need to consider
                        if dx == 0 and dy == 0:
                            continue
                        currx = x + dx
                        curry = y + dy
                        # in the image and it is foreground pixel and hasn't been labelled yet.
                        if 0 <= currx and currx < rows and 0 <= curry and curry < cols:
                            if image[currx][curry] == FOREGROUND and labelledImage[currx][curry] == 0:
                                labelledImage[currx][curry] = currLabel
                                q.put( (currx, curry) )
            
            currLabel += 1
        
    # I need to minus one to find the number since there for currLabel plus 1 at the last iteration.
    return labelledImage, currLabel-1 


def main():
    """main function"""
    image = mpimg.imread('../Q6.png')
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.show()

    binaryImage = autoThreshold(image)
    cclImage, numLabel = connected_component_labeling(binaryImage)
    print("Number of cells counted in the image is {}.".format(numLabel))
    plt.title("Labeled Image")
    plt.imshow(cclImage)
    plt.show()


if __name__ == "__main__":
    main()
    
