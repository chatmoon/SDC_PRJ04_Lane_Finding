from step0 import PARSE_ARGS
from step2 import combine_threshold
from step3 import image_warp, birds_eye_view

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import pickle


class TRACKER(object):
    def __init__(self, window_width=50,
                       nwindows=9,
                       margin=100,
                       minpix=50,
                       xm=3.7 / 812,
                       ym=10 / 720,
                       smooth_factor=15):
        self.recent_centers = []  # list that stores all the past (left, right) center set values used for smoothing the output
        self.window_width = window_width  # the window pixel width of the center values, used to count pixels inside center windows to determine curve values
        self.nwindows = nwindows  # number of sliding windows
        self.margin = margin  # set the width of the windows +/- self.margin
        self.minpix = minpix  # set minimum number of pixels found to recenter window
        self.ym_per_pix = ym  # meters per pixel in vertical axis
        self.xm_per_pix = xm  # meters per pixel in horizontal axis
        self.smooth_factor = smooth_factor

        self.timer = 5
        self.flag  = False
        self.current_fit = [np.array([False])]  # polynomial coefficients for the most recent fit

        self.left_fit  = None
        self.right_fit = None

        self.detected = False  # was the line detected in the last iteration?
        self.recent_xfitted = []  # x values of the last n fits of the line
        self.bestx = None  # average x values of the fitted line over the last n iterations
        self.best_fit = None  # polynomial coefficients averaged over the last n iterations
        
        self.radius_of_curvature = None  # radius of curvature of the line in some units
        self.line_base_pos = None  # distance in meters of vehicle center from the line
        self.diffs = np.array([0, 0, 0], dtype='float')  # difference in fit coefficients between last and new fits
        self.allx = None  # x values for detected line pixels
        self.ally = None  # y values for detected line pixels
        # self.yvals          = None
        # self.res_yvals      = None

    def find_histopeak(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Find the peak of the left and right halves of the histogram (the starting point for the left and right lines)
        midpoint = np.int(histogram.shape[0] // 2)
        (leftx_base, rightx_base) = (np.argmax(histogram[:midpoint]), np.argmax(histogram[midpoint:]) + midpoint)
        return leftx_base, rightx_base

    def find_nonzero_pixels(self, binary_warped):
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        (nonzerox, nonzeroy) = (np.array(nonzero[1]), np.array(nonzero[0]))
        return nonzerox, nonzeroy

    def search_sliding_window(self, binary_warped, leftx_current, rightx_current, window_height, nonzerox, nonzeroy,
                              out_image, to_draw=True):
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds, right_lane_inds = [], []
        # step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            (win_y_low, win_y_high) = (binary_warped.shape[0] - (window + 1) * window_height, binary_warped.shape[0] - window * window_height)
            (win_xleft_low, win_xleft_high) = (leftx_current - self.margin, leftx_current + self.margin)
            (win_xright_low, win_xright_high) = (rightx_current - self.margin, rightx_current + self.margin)
            # Draw the windows on the visualization image
            if to_draw:
                cv2.rectangle(out_image, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_image, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > self.minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        (left_lane_inds, right_lane_inds) = (np.concatenate(left_lane_inds), np.concatenate(right_lane_inds))

        # Extract left and right line pixel positions
        (leftx, lefty) = (nonzerox[left_lane_inds], nonzeroy[left_lane_inds])
        (rightx, righty) = (nonzerox[right_lane_inds], nonzeroy[right_lane_inds])

        return out_image, (leftx, lefty), (rightx, righty)

    def find_lanes_init(self, binary_warped, to_draw=False):
        # Current positions to be updated for each window
        leftx_current, rightx_current = self.find_histopeak(binary_warped)
        # Set pixel height of windows of the center value
        window_height = np.int(binary_warped.shape[0] // self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzerox, nonzeroy = self.find_nonzero_pixels(binary_warped)
        # Create an output image to draw on and  visualize the result
        out_image = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Step through the windows one by one
        out_image, (leftx, lefty), (rightx, righty) = self.search_sliding_window(binary_warped, leftx_current,rightx_current, window_height,nonzerox, nonzeroy, out_image,to_draw=to_draw)
        # Fit a second order polynomial to each
        (left_fit, right_fit) = (np.polyfit(lefty, leftx, 2), np.polyfit(righty, rightx, 2))
        # self.current_fit = [np.array([ left_fit, right_fit ])]  # self.current_fit.append([left_fit, right_fit])  
        return (left_fit, right_fit)

    def search_lane_inds(self, binary_warped, left_fit, right_fit, nonzerox, nonzeroy):
        # Identify the nonzero pixels in x and y within the window
        left_lane_inds  = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy
                          + left_fit[2] - self.margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2)
                          + left_fit[1]*nonzeroy + left_fit[2] + self.margin)))

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy
                           + right_fit[2] - self.margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2)
                           + right_fit[1]*nonzeroy + right_fit[2] + self.margin)))

        # Extract left and right line pixel positions
        ( leftx , lefty )  = ( nonzerox[left_lane_inds], nonzeroy[left_lane_inds] )
        ( rightx, righty ) = ( nonzerox[right_lane_inds], nonzeroy[right_lane_inds] )
        return ( leftx , lefty ), ( rightx, righty )

        # Extract left and right line pixel positions
        (leftx, lefty) = (nonzerox[left_lane_inds], nonzeroy[left_lane_inds])
        (rightx, righty) = (nonzerox[right_lane_inds], nonzeroy[right_lane_inds])
        return (leftx, lefty), (rightx, righty)

    def find_lanes_next(self, binary_warped):
        '''
        from the next frame of video (also called "binary_warped")
        the 1st frame should have been processed using  find_lanes_init()
        '''
        if self.flag:
            # identify the x and y positions of all nonzero pixels in the image
            nonzerox, nonzeroy = self.find_nonzero_pixels(binary_warped)
            # identify the nonzero pixels in x and y within the window
            ( leftx, lefty ), ( rightx, righty ) = self.search_lane_inds(binary_warped, self.left_fit, self.right_fit, nonzerox, nonzeroy)
            # fit a second order polynomial to each
            ( self.left_fit, self.right_fit )    = ( np.polyfit(lefty, leftx, 2), np.polyfit(righty, rightx, 2) )
        else:
            ( self.left_fit, self.right_fit )    = self.find_lanes_init(binary_warped)
            self.flag = True
        return ( self.left_fit, self.right_fit )

    def curvature(self, height, lane_x):
        x, y = lane_x[:, 0], lane_x[:, 1]  # np.linspace(0, binary_warped.shape[0]-1, num=len(lane_x))
        curve_fit_cr = np.polyfit(y * self.ym_per_pix, x * self.xm_per_pix, 2)
        return ((1 + (2 * curve_fit_cr[0] * height * self.ym_per_pix + curve_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * curve_fit_cr[0])

    def camera_offset(self, width, left_fitx, right_fitx):
        # calculate the offset of the car on the road
        camera_center = (left_fitx[-1] - right_fitx[-1]) / 2
        center_diff = (camera_center - width / 2) * self.xm_per_pix
        side_pos = 'left'
        if center_diff <= 0:
            side_pos = 'right'
        return center_diff, side_pos

    def draw_lane(self, image, left_fit, right_fit, Minv):
        # image: width, height
        (width, height) = reversed(image.shape[:2])
        # generate x and y values for plotting
        y_lane  = np.linspace(0, height-1, height )
        x_left  = left_fit[0]*y_lane**2  + left_fit[1] *y_lane + left_fit[2]
        x_right = right_fit[0]*y_lane**2 + right_fit[1]*y_lane + right_fit[2]

        left_lane  = np.column_stack(( x_left, y_lane ))
        right_lane = np.column_stack(( x_right, y_lane ))
        inner_lane = np.array(list(zip(np.concatenate((x_left + self.window_width / 2, x_right[::-1] - self.window_width / 2), axis=0),
                                       np.concatenate((y_lane, y_lane[::-1]), axis=0))), np.int32)

        # draw the lane onto the image_warped blank image
        road     = np.zeros_like(image)
        road_bkg = np.zeros_like(image)

        cv2.fillPoly(road, np.int32([left_lane]), color=[255, 0, 0])
        cv2.fillPoly(road, np.int32([right_lane]), color=[0, 0, 255])
        cv2.fillPoly(road, np.int32([inner_lane]), color=[0, 255, 0])
        cv2.fillPoly(road_bkg, np.int32([left_lane]), color=[255, 255, 255])
        cv2.fillPoly(road_bkg, np.int32([right_lane]), color=[255, 255, 255])

        road_warped = cv2.warpPerspective(road, Minv, (width, height), flags=cv2.INTER_LINEAR)
        road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, (width, height), flags=cv2.INTER_LINEAR)

        base = cv2.addWeighted(image, 1.0, road_warped_bkg, -1.0, 0.0)  # 1.3, 0.0)
        result = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)  # 1.3, 0.0)

        # calculate the offset of the car on the road
        center_diff, side_pos = self.camera_offset(width, x_left, x_right)

        # draw the text showing curvature, offset, and speed
        curverad = self.curvature(height, left_lane)
        cv2.putText(result, 'Radius of Curvature = ' + str(int(curverad)) + ' m', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, 'Vehicle is ' + str(abs(round(center_diff, 3))) + ' m ' + side_pos + ' of center', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return result


    # def draw_lane(self, image, left_lane, right_lane, inner_lane, left_fitx, right_fitx, Minv):
    #     # image: width, height
    #     (width, height) = reversed(image.shape[:2])
    #     # draw the lane onto the image_warped blank image
    #     road = np.zeros_like(image)
    #     road_bkg = np.zeros_like(image)

    #     cv2.fillPoly(road, np.int_([left_lane]), color=[255, 0, 0])
    #     cv2.fillPoly(road, np.int_([right_lane]), color=[0, 0, 255])
    #     cv2.fillPoly(road, np.int_([inner_lane]), color=[0, 255, 0])
    #     cv2.fillPoly(road_bkg, np.int_([left_lane]), color=[255, 255, 255])
    #     cv2.fillPoly(road_bkg, np.int_([right_lane]), color=[255, 255, 255])

    #     road_warped = cv2.warpPerspective(road, Minv, (width, height), flags=cv2.INTER_LINEAR)
    #     road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, (width, height), flags=cv2.INTER_LINEAR)

    #     base = cv2.addWeighted(image, 1.0, road_warped_bkg, -1.0, 0.0)  # 1.3, 0.0)
    #     result = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)  # 1.3, 0.0)

    #     # calculate the offset of the car on the road
    #     center_diff, side_pos = self.camera_offset(width, left_fitx, right_fitx)

    #     # draw the text showing curvature, offset, and speed
    #     curverad = self.curvature(height, left_lane)
    #     cv2.putText(result, 'Radius of Curvature = ' + str(int(curverad)) + ' m', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #                 (255, 255, 255), 2)
    #     cv2.putText(result, 'Vehicle is ' + str(abs(round(center_diff, 3))) + ' m ' + side_pos + ' of center',
    #                 (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    #     return result

    # def fit_lanes(self, binary_warped, leftx, rightx):
    #     # Set pixel height of windows of the center value
    #     window_height = np.int(binary_warped.shape[0] // self.nwindows)
    #     # fit the lane boundaries to the left, right center positions found
    #     ( yvals, res_yvals )      = ( range(0, binary_warped.shape[0]), np.arange( binary_warped.shape[0] - (window_height/2), 0, -window_height     ) )
    #     ( left_fit, left_fitx )   = ( np.polyfit(res_yvals, leftx, 2) , np.array( left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2], np.int32   ) )
    #     ( right_fit, right_fitx ) = ( np.polyfit(res_yvals, rightx, 2), np.array( right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2], np.int32) )
    #     left_lane  = np.array(list(zip(np.concatenate(( left_fix - self.window_width/2, left_fix[::-1] + self.window_width/2), axis=0), np.concatenate(( yvals,yvals[::-1]), axis=0))), np.int32)
    #     right_lane = np.array(list(zip(np.concatenate(( right_fix - self.window_width/2, right_fix[::-1] + self.window_width/2), axis=0), np.concatenate(( yvals,yvals[::-1]), axis=0))), np.int32)
    #     inner_lane = np.array(list(zip(np.concatenate(( left_fitx + self.window_width / 2, right_fitx[::-1] - self.window_width / 2), axis=0), np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    #     return ( left_fitx, right_fitx ), ( yvals, res_yvals ), ( left_lane, right_lane, inner_lane )

    # def find_lanes_init(self, binary_warped, to_draw=False):
    #     # Current positions to be updated for each window
    #     leftx_current, rightx_current = self.find_histopeak(binary_warped)
    #     # Set pixel height of windows of the center value
    #     window_height = np.int(binary_warped.shape[0] // self.nwindows)
    #     # Identify the x and y positions of all nonzero pixels in the image
    #     nonzerox, nonzeroy = self.find_nonzero_pixels(binary_warped)
    #     # Create an output image to draw on and  visualize the result
    #     out_image = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    #     # Step through the windows one by one
    #     out_image, (leftx, lefty), (rightx, righty) = self.search_sliding_window(binary_warped, leftx_current,rightx_current, window_height,nonzerox, nonzeroy, out_image,to_draw=to_draw)
    #     # Fit a second order polynomial to each
    #     (left_fit, right_fit) = (np.polyfit(lefty, leftx, 2), np.polyfit(righty, rightx, 2))
    #     return (left_fit, right_fit)

    # def find_lanes_next(self, binary_warped, left_fit, right_fit):
    #     '''
    #     from the next frame of video (also called "binary_warped")
    #     the 1st frame should have been processed using  find_lanes_init()
    #     '''
    #     # Identify the x and y positions of all nonzero pixels in the image
    #     nonzerox, nonzeroy = self.find_nonzero_pixels(binary_warped)
    #     # Identify the nonzero pixels in x and y within the window
    #     ( leftx, lefty ), ( rightx, righty ) = self.search_lane_inds(binary_warped, left_fit, right_fit, nonzerox, nonzeroy)
    #     # fit a second order polynomial to each
    #     ( left_fit, right_fit ) = ( np.polyfit(lefty, leftx, 2), np.polyfit(righty, rightx, 2) )

    #     return ( left_fit, right_fit )



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