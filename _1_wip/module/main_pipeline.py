# import modules - see goo.gl/oe6jZL
from step0 import PARSE_ARGS
from stepA import camera_calibrate
from stepB import *
from stepC import *
from check import *
from tracker import TRACKER #, window_mask, search_sliding_window
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from math import isclose
import pandas as pd


# parameter
directory = 'D:/USER/_PROJECT_/_PRJ04_/_1_WIP/_1_forge/_4_v3/'
args = PARSE_ARGS(path=directory)

if os.path.exists(args.cali + 'calibration.p'):
    # read in the saved objpoints and imgpoints
    camera = pickle.load(open(args.cali + "calibration.p", "rb"))
else:
    # return the camera matrix, distortion coefficients
    camera = camera_calibrate(args)
    # save the camera calibration result
    pickle.dump(camera, open(args.cali + 'calibration.p', 'wb'))
mtx    = camera['camera_matrix']
dist   = camera['coef_distorsion']

global d
d = []

def process_image(image):
    # image: width, height
    (width, height) = reversed(image.shape[:2])
    # undistort image
    image = cv2.undistort(image, mtx, dist, None, mtx)
    # process image, generate binary pixel of interest
    image_binary = combine_threshold(args, image, stacked_image=False, region=False) # = 1 <- replace by '=255'
    # perform the perspective transform
    M, Minv, image_warped = image_warp(args, image_binary)

    # find lane line
    (window_width, window_height) = (25, 80) # (50, 80) (25, 80)
    (margin, smooth_factor)       = (100, 10) # (100, 10) (100, 15) (30, 20) (100, 20)
    (xm, ym)                      = (3.7/812, 10/720) # (1, 1) (3.7/783, 10/720)
    (yvals, res_yvals)            = np.array(range(0, height)), np.arange(height - (window_height / 2), 0, -window_height)

    # set up the overall class to do all the tracking:
    curve_centers    = TRACKER(Mywindow_width=window_width, Mywindow_height=window_height, Mymargin=margin, My_ym=ym, My_xm=xm, Mysmooth_factor=smooth_factor) # (Mywindow_width=window_width, Mywindow_height=window_height)
    window_centroids = curve_centers.find_window_centroids(image_warped)

    (leftx, rightx) = [ window_centroids[:, i] for i in range(2)]
    (left_lane , left_fitx , coeff_left), (right_lane, right_fitx, coeff_right), inner_lane = curve_centers.fit_lanes(window_centroids)

    # TODO: add sanity check here
    check1 = check_grad_lane(left_lane, right_lane, xm_per_pix=xm, clearance=0.3, trigger=0.65)
    check2 = isclose(curve_centers.curvature(height, left_lane), curve_centers.curvature(height, right_lane), abs_tol=10)
    if check1 and check2: #if check_grad_lane_0(left_lane,right_lane, clearance=20):
        d.append( {'left_lane': left_lane, 'left_fitx': left_fitx, 'coeff_left': coeff_left,
                   'right_lane': right_lane, 'right_fitx': right_fitx, 'coeff_right': coeff_right, 'inner_lane': inner_lane} )
        return curve_centers.draw_lane(image, window_centroids, left_lane, right_lane, inner_lane, left_fitx, right_fitx, Minv)
    else:
        if len(d) != 0:
            return curve_centers.draw_lane(image, window_centroids, d[-1]['left_lane'], d[-1]['right_lane'], d[-1]['inner_lane'], d[-1]['left_fitx'], d[-1]['right_fitx'], Minv)
        else:
            return image

    #return curve_centers.draw_lane(image, window_centroids, left_lane, right_lane, inner_lane, left_fitx, right_fitx, Minv)

# video code
def video(video_input, video_output):
    clip1      = VideoFileClip(video_input)
    video_clip = clip1.fl_image(process_image)
    #% time white_clip.write_videofile(video_output, audio=False)
    video_clip.write_videofile(video_output, audio=False)

def test(args, mp4=0):
    video_input  = args.path + {0: "project_video.mp4", 1: "challenge_video.mp4", 2: "harder_challenge_video.mp4"}[mp4]
    video_output = args.out  + {0: "video_output.mp4", 1: "video_output_challenge.mp4", 2: "video_output_harder.mp4"}[mp4]
    video(video_input, video_output)

def main():
    # parameter
    directory = 'D:/USER/_PROJECT_/_PRJ04_/_1_WIP/_1_forge/_4_v3/'
    args = PARSE_ARGS(path=directory)

    # generate video output
    test(args, mp4=0)
    # test(args, mp4=1)
    # test(args, mp4=2)

if __name__ == '__main__':
    main()