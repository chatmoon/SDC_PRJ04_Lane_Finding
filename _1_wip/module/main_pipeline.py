from step0 import PARSE_ARGS
from step1 import camera_calibrate
from step2 import combine_threshold
from step3 import image_warp, birds_eye_view
from tracker import TRACKER
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from math import isclose
import pandas as pd
global d
d = []

# parameter
#directory = 'D:/USER/_PROJECT_/_PRJ04_/_1_WIP/_1_forge/_5_v4/'
directory = 'D:/USER/_PROJECT_/_PRJ04_/_1_WIP/_2_github/_1_wip/'
args      = PARSE_ARGS(path=directory)

# return the camera matrix, distortion coefficients
camera = camera_calibrate(args)
mtx    = camera['camera_matrix']
dist   = camera['coef_distorsion']

def process_image(image):
    # image: width, height
    (width, height) = reversed(image.shape[:2])
    # undistort image
    image = cv2.undistort(image, mtx, dist, None, mtx)
    # process image, generate binary pixel of interest
    image_binary = combine_threshold(args, image, stacked_image=False, region=False) # = 1 <- replace by '=255'
    # perform the perspective transform
    M, Minv, image_warped = image_warp(args, image_binary)

    # find lane lines
    # (yvals, res_yvals) = np.array(range(0, height)), np.arange(height - (window_height / 2), 0, -window_height)
    line = TRACKER(window_width=25,nwindows= 9, margin= 100, minpix= 50, xm= 3.7/812, ym= 10/720, smooth_factor= 15)
    ( left_fit, right_fit ) = line.find_lanes_next(image_warped)


    
    # # TODO: add sanity check here
    # check0 = len(leftx)
    # check1 = check_grad_lane(left_lane, right_lane, xm_per_pix=3.7/812, clearance=0.3, trigger=0.65)
    # check2 = isclose(curve_centers.curvature(height, left_lane), curve_centers.curvature(height, right_lane), abs_tol=10)
    # if check1 and check2: #if check_grad_lane_0(left_lane,right_lane, clearance=20):
    #     d.append( {'left_lane': left_lane, 'left_fitx': left_fitx, 'coeff_left': coeff_left,
    #                'right_lane': right_lane, 'right_fitx': right_fitx, 'coeff_right': coeff_right, 'inner_lane': inner_lane} )
    #     return curve_centers.draw_lane(image, left_lane, right_lane, inner_lane, left_fitx, right_fitx, Minv)
    # else:
    #     if len(d) != 0:
    #         return curve_centers.draw_lane(image, d[-1]['left_lane'], d[-1]['right_lane'], d[-1]['inner_lane'], d[-1]['left_fitx'], d[-1]['right_fitx'], Minv)
    #     else:
    #         return image

    return line.draw_lane(image, left_fit, right_fit, Minv)

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
    # # parameter
    # directory = 'D:/USER/_PROJECT_/_PRJ04_/_1_WIP/_1_forge/_5_v4/'
    # args = PARSE_ARGS(path=directory)

    # generate video output
    test(args, mp4=0)
    # test(args, mp4=1)
    # test(args, mp4=2)

if __name__ == '__main__':
    main()