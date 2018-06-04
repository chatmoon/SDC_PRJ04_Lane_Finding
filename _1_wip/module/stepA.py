from step0 import PARSE_ARGS
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import pickle


# Helper function: camera calibration
def camera_calibrate(args):
    # prepare object points
    nx, ny = args.incorner #number of inside corners in x, in y
    # read in a calibration image
    images = glob.glob(args.cali + '*.jpg')
    # arrays to store object points and image points from all the images
    output = {}
    output['objpoints'], output['imgpoints'] = [], [] # 3D points in real world space, 2D points in image plane
    output['images_corners_found'], output['images_corners_not_found']  = [], [] # list of image names
    output['ret'] = [] # list of True and False

    # prepare object points, like (0,0,0) , (1,0,0) , (2,0,0) ..., (7,5,0)
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # x, y coordinates

    for frame in images:
        # read in a calibration frame
        image = mpimg.imread(frame) # RGB for moviepy
        # convert frame to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # find the chessboard corners #output['ret'], output['corners'] = cv2.findChessboardCorners(gray, (nx, ny), None)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        output['ret'].append(ret)
        # if corners are found, add object points, frame points
        if ret == True:
            output['imgpoints'].append(corners)
            output['objpoints'].append(objp)
            output['images_corners_found'].append(frame)
        else:
            output['images_corners_not_found'].append(frame)
    reprojection_error, output['camera_matrix'], output['coef_distorsion'], rvecs, tvecs = cv2.calibrateCamera(output['objpoints'], output['imgpoints'], gray.shape[::-1], None, None)

    return output


def main():
    '''
    LAYOUT: camera calibration
     . calculate mxt, dist
     . save it on the hard drive (pickle)
    '''
    # parameter
    directory = 'D:/USER/_PROJECT_/_PRJ04_/_1_WIP/_1_forge/_4_v3/'
    args = PARSE_ARGS(path=directory)
    # return the camera matrix, distortion coefficients
    camera_dictionary = camera_calibrate(args)
    # save the camera calibration result
    pickle.dump(camera_dictionary, open(args.cali + 'calibration.p', 'wb'))


if __name__ == '__main__':
    main()