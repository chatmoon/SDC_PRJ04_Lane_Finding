from step0 import PARSE_ARGS
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import pickle


# Helper functions: define perspective transform function
def image_warp(args, image, x_top = [ 594, 686 ]): # [586, 692]):
    # define calibration box in source (original) and destination (desired and wraped) coordinates
    (width, height) = reversed(image.shape[:2])  # img.shape[:2][::-1]
    # four source coordinates
    src = np.float32([ [x_top[0], 450], [x_top[1], 450], [1045, 665], [262, 665]])
    # four desired coordinates
    #dst = np.float32([[50, 35] , [50, 115] , [230, 115] , [230, 35]]) # dst1
    #dst = np.float32([[args.offset, args.offset], [height - args.offset, args.offset], [height - args.offset, width - args.offset], [args.offset, width - args.offset]])  # dst2
    #dst = np.float32([[400, 0], [900, 0], [900, 720], [400, 720]]) # dst3
    #dst = np.float32([[200, 0], [1100, 0], [1100, 720], [200, 720]]) # dst4
    dst = np.float32([ [262, 100], [1045, 100], [1045, 665], [262, 665] ])  # dst5
    # compute the perspective transform M
    M = cv2.getPerspectiveTransform(src, dst)
    # # compute the inverse also by swapping the input parameters
    Minv = cv2.getPerspectiveTransform(dst, src)
    # create warped image - uses linear interpolation
    warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
    return M, Minv, warped

def main():
    pass

if __name__ == '__main__':
    main()