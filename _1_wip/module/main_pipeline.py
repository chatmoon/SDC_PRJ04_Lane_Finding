# import modules - see goo.gl/oe6jZL
from step0 import PARSE_ARGS
from stepA import camera_calibrate
from stepB import *
from stepC import *
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
import pandas as pd


# parameter
directory = 'D:/USER/_PROJECT_/_PRJ04_/_1_WIP/_2_github/_1_wip/'
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


def check_shift_x(lane, clearance=30):
    a0 = [lane[i] - lane[i + 1] for i in range(len(lane) - 1)]  # diff between x(i) & x(i+1)
    a1 = [0 if (np.sign(a0[i]) == np.sign(a0[i + 1]) or a0[i] == 0) or abs(a0[i + 1]) < clearance else 1 for i in range(len(a0) - 1)]
    if sum(a1) == 0:
        return True
    else:
        return False

def check_shift_lane_width(wc, clearance=200):
    a = [wc[i][1] - wc[i][0] for i in range(len(wc))]
    b = a - min(a)
    if max(b) < clearance:
        return True
    else:
        return False


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
    (xm, ym)                      = (3.7/783, 10/720) # (1, 1) (3.7/783, 10/720)
    (yvals, res_yvals)            = np.array(range(0, height)), np.arange(height - (window_height / 2), 0, -window_height)

    # set up the overall class to do all the tracking:
    curve_centers    = TRACKER(Mywindow_width=window_width, Mywindow_height=window_height, Mymargin=margin, My_ym=ym, My_xm=xm, Mysmooth_factor=smooth_factor) # (Mywindow_width=window_width, Mywindow_height=window_height)
    window_centroids = curve_centers.find_window_centroids(image_warped)

    (leftx, rightx) = [ window_centroids[:, i] for i in range(2)]
    (left_lane , left_fitx , coeff_left), (right_lane, right_fitx, coeff_right), inner_lane = curve_centers.fit_lanes(window_centroids)

    # TODO: add sanity check here
    if 1 == 1:
        pass

    return curve_centers.draw_lane(image, window_centroids, left_lane, right_lane, inner_lane, left_fitx, right_fitx, Minv)

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