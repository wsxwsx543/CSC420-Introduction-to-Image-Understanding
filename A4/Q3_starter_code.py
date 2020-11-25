import numpy as np
import cv2
import plotly.graph_objects as go


def get_data(folder):
    '''
    reads data in the specified image folder
    '''
    depth = cv2.imread(folder + 'depthImage.png')[:,:,0]
    rgb = cv2.imread(folder + 'rgbImage.jpg')
    extrinsics = np.loadtxt(folder + 'extrinsic.txt')
    intrinsics = np.loadtxt(folder + 'intrinsics.txt')
    return depth, rgb, extrinsics, intrinsics



def compute_point_cloud(imageNumber):
    '''
     This function provides the coordinates of the associated 3D scene point
     (X; Y;Z) and the associated color channel values for any pixel in the
     depth image. You should save your output in the output_file in the
     format of a N x 6 matrix where N is the number of 3D points with 3
     coordinates and 3 color channel values:
     X_1,Y_1,Z_1,R_1,G_1,B_1
     X_2,Y_2,Z_2,R_2,G_2,B_2
     X_3,Y_3,Z_3,R_3,G_3,B_3
     X_4,Y_4,Z_4,R_4,G_4,B_4
     X_5,Y_5,Z_5,R_5,G_5,B_5
     X_6,Y_6,Z_6,R_6,G_6,B_6
     .
     .
     .
     .
    '''
    depth, rgb, extrinsics, intrinsics = get_data(imageNumber)
    # rotation matrix
    R = extrinsics[:, :3]
    # t
    t = extrinsics[:, 3]

# YOUR IMPLEMENTATION CAN GO HERE:


    return results


def plot_pointCloud(pc):
    '''
    plots the Nx6 point cloud pc in 3D
    assumes (1,0,0), (0,1,0), (0,0,-1) as basis
    '''
    fig = go.Figure(data=[go.Scatter3d(
        x=pc[:, 0],
        y=pc[:, 1],
        z=-pc[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=pc[:, 3:][..., ::-1],
            opacity=0.8
        )
    )])
    fig.show()



if __name__ == '__main__':

    imageNumbers = ['1/', '2/', '3/']
    for  imageNumber in  imageNumbers:

        # Part a)
        pc = compute_point_cloud( imageNumber)
        np.savetxt( imageNumber + 'pointCloud.txt', pc)
        plot_pointCloud(pc)

