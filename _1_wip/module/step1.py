from step0 import PARSE_ARGS
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import glob
import pickle


# Helper function: camera calibration
def camera_calibrate(args):
    if os.path.exists(args.cali + 'calibration.p'):
        # read in the saved objpoints and imgpoints
        camera = pickle.load(open(args.cali + "calibration.p", "rb"))
    else:
        # prepare object points
        nx, ny = args.incorner #number of inside corners in x, in y
        # read in a calibration image
        images = glob.glob(args.cali + '*.jpg')
        # arrays to store object points and image points from all the images
        camera = {}
        camera['objpoints'], camera['imgpoints'] = [], [] # 3D points in real world space, 2D points in image plane
        camera['images_corners_found'], camera['images_corners_not_found']  = [], [] # list of image names
        camera['ret'] = [] # list of True and False

        # prepare object points, like (0,0,0) , (1,0,0) , (2,0,0) ..., (7,5,0)
        objp = np.zeros((nx*ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # x, y coordinates

        for frame in images:
            # read in a calibration frame
            image = mpimg.imread(frame) # RGB for moviepy
            # convert frame to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # find the chessboard corners #camera['ret'], camera['corners'] = cv2.findChessboardCorners(gray, (nx, ny), None)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            camera['ret'].append(ret)
            # if corners are found, add object points, frame points
            if ret == True:
                camera['imgpoints'].append(corners)
                camera['objpoints'].append(objp)
                camera['images_corners_found'].append(frame)
            else:
                camera['images_corners_not_found'].append(frame)
                
        reprojection_error, camera['camera_matrix'], camera['coef_distorsion'], rvecs, tvecs = cv2.calibrateCamera(camera['objpoints'], camera['imgpoints'], gray.shape[::-1], None, None)

    return camera


#Helper function: draw the corners into the camera calibration images
def corners_draw(args, camera_dictionary):
    nx, ny = args.incorner
    image_to_draw = []
    images, imgpoints, rets = camera_dictionary['images_corners_found'], camera_dictionary['imgpoints'], camera_dictionary['ret']
    for frame, corners, ret in zip(images, imgpoints, rets):
        # read in a calibration frame
        image = mpimg.imread(frame)
        # draw and display the corners
        image = cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
        image_to_draw.append(image)
    return image_to_draw


# Helper functions: undistort image
def image_undistort(args, image):
    # return the camera matrix, distortion coefficients
    output = camera_calibrate(args)
    # undistort using camera_matrix and coef_distorsion
    return  cv2.undistort(image, output['camera_matrix'], output['coef_distorsion'], None, output['camera_matrix'])


# Helper function: plot images
def images_plot(args, camera_dictionary):
    fig, ax_lanes = plt.subplots(figsize=(15,13), nrows=5, ncols=1)
    string_list = [' ', ' not']

    for row, ax_lane in enumerate(ax_lanes, start=1):
        if row == 1 or row == 5:
            ax_lane.set_title('{}. the camera calibration images for which the corners were{} found\n'.format(row%3, string_list[row%3 - 1]), fontsize=20)
        # Turn off axis lines and ticks of the lane subplot
        ax_lane.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        # remove the white frame
        ax_lane._frameon = False

    # load images with/without drawn corners
    images_corners_found     = corners_draw(args, camera_dictionary)
    images_corners_not_found = [mpimg.imread(image) for image in camera_dictionary['images_corners_not_found']]
    len_found, len_not_found = len(images_corners_found), len(images_corners_not_found)
    size_x, size_y, channel = images_corners_found[0].shape
    image_white    = np.zeros([size_x, size_y,3],dtype=np.uint8)
    image_white.fill(255)
    
    
    for i in range(1, 1 + 25):
        # select where to plot the image in the grid
        ax_image = fig.add_subplot(5, args.column, i)

        if   i in range(1, 1+17): # images_corners_found
            offset    = 0
            image     = images_corners_found[(i-1)]
            title     = camera_dictionary['images_corners_found'][(i-1)].split('\\')[-1]
            #image_qty = len_found
        elif i in range(1+17, 21):
            image     = image_white
            title     = ''
            #image_qty = len_found
        elif i in range(21, 1+23): # images_corners_not_found
            offset    = 20
            image     = images_corners_not_found[(i-1)-offset]
            title     = camera_dictionary['images_corners_not_found'][(i-1)-offset].split('\\')[-1]
            #image_qty = len_not_found
        else:
            image     = image_white
            title     = ''
            #image_qty = len_not_found   
        ax_image.imshow(image)
        ax_image.axis('off')

        ax_image.set_title(title, fontsize=16)
        ax_image.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        plt.imshow(image)

    plt.tight_layout()
    plt.show()


# Helper function: plot images
def images_show(args, image_to_plot):
    # plot images
    remainder = len(image_to_plot) % args.column
    iquotient = len(image_to_plot) // args.column
    rows = iquotient if remainder == 0 else 1 + iquotient

    figure, axes = plt.subplots(rows, args.column, figsize=args.figsize) # (15, 13)
    w = rows * args.column - len(image_to_plot)
    _ = [axes[-1, -i].axis('off') for i in range(1, w + 1)]
    figure.tight_layout()

    flag = True
    for ax, image in zip(axes.flatten(), image_to_plot):
        if flag:
            ax.imshow(image[1])
            flag = False
        else:
            ax.imshow(image[1], cmap='gray')
        ax.set_title(image[0], fontsize=15)

        if args.to_plot:
            ax.grid(color='y', linestyle='-', linewidth=1)

    plt.show()


def main():
    '''
    LAYOUT: camera calibration
     . calculate mxt, dist
     . save it on the hard drive (pickle)
     . plot images
    '''
    # parameter
    directory = 'D:/USER/_PROJECT_/_PRJ04_/_1_WIP/_1_forge/_5_v4/'
    args = PARSE_ARGS(path=directory)
    # return the camera matrix, distortion coefficients
    camera_dictionary = camera_calibrate(args)
    # save the camera calibration result
    pickle.dump(camera_dictionary, open(args.cali + 'calibration.p', 'wb'))
    # plot the camera calibration result
    images_plot(args, camera_dictionary)

if __name__ == '__main__':
    main()