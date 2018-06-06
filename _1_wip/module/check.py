import numpy as np
import cv2
import csv
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def check_shift_x(lane, clearance=20):
    a0 = [lane[i] - lane[i + 1] for i in range(len(lane) - 1)]  # diff between x(i) & x(i+1)
    a1 = [0 if (np.sign(a0[i]) == np.sign(a0[i + 1]) or a0[i] == 0) and abs(a0[i + 1]) < clearance else 1 for i in range(len(a0) - 1)]
    if sum(a1) == 0:
        return True
    else:
        return False

def check_grad_lane(left_lane,right_lane, xm_per_pix, clearance=0.3, trigger=0.5):
    a = (right_lane[:, 0] - left_lane[:, 0])*xm_per_pix
    b = np.sum( (a > 3.7 - clearance ) & (a < 3.7 + clearance))
    c = b/len(a)
    if c >= trigger:
        return True
    else:
        return False

def check_curve_parallel():
    '''
    1. select one point, pt_left_i, from the left lane
    2. get the normal to the left curve at this point
    3. find the intersection between the normal and the right lane, pt_right_i
    4. compare the tangent of the both curves, at the two points, pt_left_i and pt_right_i
    5. if the tangents are similar, then the curves are parallel at these points
    '''
    pass

# ----------------------------------------------------------------
def check_shift_lane_width(wc, clearance=200):
    a = [wc[i][1] - wc[i][0] for i in range(len(wc))]
    b = a - min(a)
    if max(b) < clearance:
        return True
    else:
        return False

def check_grad_lane_0(left_lane,right_lane, clearance=20):
    a = np.gradient(np.gradient([left_lane[:, 0], right_lane[:, 0]], axis=0), axis=1)
    b = max(abs(a[0]))
    if b < clearance:
        return True
    else:
        return False

def check_grad_wc(window_centroids, clearance=20):
    a = np.gradient(window_centroids, axis=1)[:, 0]
    b = max(abs(np.gradient(a)))
    if b < clearance:
        return True
    else:
        return False


'''
#DONE: Checking that they have similar curvature
#DONE: Checking that they are separated by approximately the right distance horizontally
* Checking that they are roughly parallel:
  . so derivative in two points have to be about the same
  . I used this difference of derivatives as a sanity check
* If the change in coefficient is above 0.005, the lanes are discarded
* Outlier removal 2:
  . If any lane was found with less than 5 pixels, we use the previous line fit coefficients as the coefficients for the current one.
Smoothing: We smooth the value of the current lane using a first order filter response, as \\(coeffs = 0.95*coeff~prev+ 0.05 coeff\\).


'''