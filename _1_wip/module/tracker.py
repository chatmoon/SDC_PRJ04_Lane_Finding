#from module.step2 import grayscale, sobel_xy, sobel_abs, scale, mask, gradient_sobel_abs, gradient_magnitude, gradient_direction
#from module.step3 import threshold_color_gray, threshold_color_rgb, threshold_color_hls, threshold_color_hsv, region_of_interest, combine_threshold
#from module.step4 import image_warp, birds_eye_view, test_birds_eye_view1, test_birds_eye_view2
#from debug import debug_window_centroids_save, debug_window_centroids_plot_
import numpy as np
import cv2
import csv
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

class TRACKER(object):
    '''
    Ref:
    - www.youtube.com/watch?v=vWY8YUayf9Q&feature=youtu.be
    - SDCNP, Lesson 15 Advanced Lane Finding, "34. Sliding Window Search"
    '''
    def __init__(self, Mywindow_width = 50, Mywindow_height = 80, Mymargin = 100, My_ym = 10/720, My_xm = 3.7/812, Mysmooth_factor = 15):
        self.recent_centers = []                # list that stores all the past (left, right) center set values used for smoothing the output
        self.window_width   = Mywindow_width    # the window pixel width of the center values, used to count pixels inside center windows to determine curve values
        self.window_height  = Mywindow_height   # the window pixel height of the center values, used to count pixels inside center windows to determine curve values
        self.margin         = Mymargin          # the pixel distance in both directions to slide (left_window + right_window) template for searching
        self.ym_per_pix     = My_ym             # meters per pixel in vertical axis
        self.xm_per_pix     = My_xm             # meters per pixel in horizontal axis
        self.smooth_factor  = Mysmooth_factor

        self.timer          = 5

        self.detected       = False             # was the line detected in the last iteration?        
        self.recent_xfitted = []                # x values of the last n fits of the line        
        self.bestx          = None              # average x values of the fitted line over the last n iterations        
        self.best_fit       = None              # polynomial coefficients averaged over the last n iterations        
        self.current_fit    = [np.array([False])]  # polynomial coefficients for the most recent fit        
        self.radius_of_curvature = None         # radius of curvature of the line in some units        
        self.line_base_pos  = None              # distance in meters of vehicle center from the line        
        self.diffs          = np.array([0,0,0], dtype='float')   # difference in fit coefficients between last and new fits        
        self.allx           = None              # x values for detected line pixels        
        self.ally           = None              # y values for detected line pixels
        #self.yvals          = None
        #self.res_yvals      = None


    def check_shift_x(self, lane, clearance=20):
        a0 = [lane[i] - lane[i + 1] for i in range(len(lane) - 1)]  # diff between x(i) & x(i+1)
        a1 = [0 if (np.sign(a0[i]) == np.sign(a0[i + 1]) or a0[i] == 0) and abs(a0[i + 1]) < clearance else 1 for i in range(len(a0) - 1)]
        if sum(a1) == 0:
            return True
        else:
            return False

    def check_shift_lane_width(self, wc, clearance=200):
        a = [wc[i][1] - wc[i][0] for i in range(len(wc))]
        b = a - min(a)
        if max(b) < clearance:
            return True
        else:
            return False

    def check_grad_wc(self, window_centroids, clearance=20):
        a = np.gradient(window_centroids, axis=1)[:, 0]
        b = max(abs(np.gradient(a)))
        if b < clearance:
            return True
        else:
            return False

    def find_window_centroids(self, image):
        window_width  = self.window_width
        window_height = self.window_height
        margin        = self.margin

        window_centroids = []  # Store the (left,right) window centroid positions per level
        window = np.ones(window_width)  # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum    = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
        r_sum    = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)

        # Add what we found for the first layer
        window_centroids.append([l_center, r_center])

        # Go through each layer looking for max pixel locations
        for level in range(1, (int)(image.shape[0] / window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width / 2
            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, image.shape[1]))
            l_center    = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, image.shape[1]))
            r_center    = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
            # Add what we found for that layer
            window_centroids.append([l_center, r_center])

        # store the (left, right) center set values and return averaged values of line centers
        check1 = len(window_centroids) > 0 #print(np.array(window_centroids)[:,0])

        count = 0
        if count > self.timer:
            if check1:
                self.recent_centers.append(window_centroids)
        else:
            self.recent_centers.append(window_centroids)
            count += 1

        # debug_window_centroids_save()
        self.bestx = np.average(self.recent_centers[ -self.smooth_factor:], axis = 0)

        return self.bestx


    def window_mask(self, image_ref, center, level): # last version: _180527-1625_tracker.py
        output = np.zeros_like(image_ref)
        output[int(image_ref.shape[0] - (level + 1) * self.window_height):int(image_ref.shape[0] - level * self.window_height),
        max(0, int(center - self.window_width / 2)):min(int(center + self.window_width / 2), image_ref.shape[1])] = 1
        return output


    def search_sliding_window(self, warped): # , curve_centers):
        # find window centroids
        window_centroids = self.find_window_centroids(warped)

        if len(window_centroids) > 0:  # If we found any window centers
            # Points used to draw all the left and right windows
            l_points = np.zeros_like(warped)
            r_points = np.zeros_like(warped)

            # Go through each level and draw the windows
            for level in range(0, len(window_centroids)):
                # Window_mask is a function to draw window areas
                l_mask = self.window_mask(self.window_width, self.window_height, warped, window_centroids[level][0], level)
                r_mask = self.window_mask(self.window_width, self.window_height, warped, window_centroids[level][1], level)
                # Add graphic points from window mask here to total pixels found
                l_points[(l_points == 255) | ((l_mask == 1))] = 255
                r_points[(r_points == 255) | ((r_mask == 1))] = 255

            # Draw the results
            template     = np.array(r_points + l_points, np.uint8)          # add both left and right window pixels together
            zero_channel = np.zeros_like(template)                          # create a zero color channel
            template     = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
            warpage      = np.dstack((warped, warped, warped)) * 255        # making the original road pixels 3 color channels
            output       = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

        else:  # If no window centers found, just display orginal road image
            output       = np.array(cv2.merge((warped, warped, warped)), np.uint8)

        return output


    def fit_lanes_sub(self, yvals, res_yvals, side_x):
        side_fit   = np.polyfit(res_yvals, side_x, 2)
        side_fitx  = side_fit[0] * yvals**2 + side_fit[1] * yvals + side_fit[2]
        side_fitx  = np.array(side_fitx, np.int32)
        side_lane  = np.array(list(zip(np.concatenate((side_fitx - self.window_width / 2, side_fitx[::-1] + self.window_width / 2), axis=0),
                                       np.concatenate((yvals, yvals[::-1]), axis=0)) ),np.int32)
        return side_lane, side_fitx, side_fit 


    def fit_lanes(self, window_centroids):
        # fit the lane boundaries to the left, right center positions found
        (leftx, rightx) = [ window_centroids[:, i] for i in range(2)]       
        yvals     = np.linspace(0, 720, num=len(leftx))
        res_yvals = np.arange(720 - (self.window_height / 2), 0, -self.window_height)

        left_lane , left_fitx , coeff_left  = self.fit_lanes_sub(yvals, res_yvals, leftx)
        right_lane, right_fitx, coeff_right = self.fit_lanes_sub(yvals, res_yvals, rightx)

        inner_lane = np.array(list(zip(np.concatenate((left_fitx + self.window_width / 2, right_fitx[::-1] - self.window_width / 2), axis=0),
                                       np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

        return (left_lane , left_fitx , coeff_left), (right_lane, right_fitx, coeff_right), inner_lane


    def curvature(self, height, lane_x):
        x, y = lane_x[:,0], lane_x[:,1] # np.linspace(0, height-1, num=len(lane_x))
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


    def draw_lane(self, image, window_centroids, left_lane, right_lane, inner_lane, left_fitx, right_fitx, Minv):
        # image: width, height
        (width, height) = reversed(image.shape[:2])
        (leftx, rightx) = [ window_centroids[:, i] for i in range(2)]  
        # draw the lane onto the image_warped blank image
        road = np.zeros_like(image)
        road_bkg = np.zeros_like(image)

        cv2.fillPoly(road, np.int_([left_lane]), color=[255, 0, 0])
        cv2.fillPoly(road, np.int_([right_lane]), color=[0, 0, 255])
        cv2.fillPoly(road, np.int_([inner_lane]), color=[0, 255, 0])
        cv2.fillPoly(road_bkg, np.int_([left_lane]), color=[255, 255, 255])
        cv2.fillPoly(road_bkg, np.int_([right_lane]), color=[255, 255, 255])
       
        road_warped     = cv2.warpPerspective(road, Minv, (width, height), flags=cv2.INTER_LINEAR)
        road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, (width, height), flags=cv2.INTER_LINEAR)

        base   = cv2.addWeighted(image, 1.0, road_warped_bkg, -1.0, 0.0)  # 1.3, 0.0)
        result = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)  # 1.3, 0.0)

        # calculate the offset of the car on the road
        center_diff, side_pos = self.camera_offset(width, left_fitx, right_fitx)

        # draw the text showing curvature, offset, and speed
        curverad = self.curvature(height, left_lane)
        cv2.putText(result, 'Radius of Curvature = ' + str(int(curverad)) + ' m', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, 'Vehicle is ' + str(abs(round(center_diff, 3))) + ' m ' + side_pos + ' of center', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return result

    # def curvature(self, height, lane_x):
    #     x, y = lane_x, np.linspace(0, height - 1, num=len(lane_x))
    #     curve_fit_cr = np.polyfit(y * self.ym_per_pix, x * self.xm_per_pix, 2)
    #     return ((1 + (2 * curve_fit_cr[0] * height * self.ym_per_pix + curve_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * curve_fit_cr[0])

    # def curvature(self, inner_lane):
    #     dx_dt = np.gradient(inner_lane[:, 0]* self.xm_per_pix)
    #     dy_dt = np.gradient(inner_lane[:, 1]* self.ym_per_pix)
    #     d2x_dt2 = np.gradient(dx_dt)
    #     d2y_dt2 = np.gradient(dy_dt)
    #     curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
    #     return curvature

    # def curvature(self, height, window_centroids):
    #     yvals     = np.array(range(0, height))
    #     res_yvals = np.arange(height - (self.window_height / 2), 0, -self.window_height)
    #     curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32) * self.ym_per_pix, np.array(window_centroids[:,0], np.float32) * self.xm_per_pix, 2)
    #     curverad     = ((1 + (2 * curve_fit_cr[0] * yvals[-1] * self.ym_per_pix + curve_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * curve_fit_cr[0])


def main():
    # parameters
    args = PARSE_ARGS()
    # list the images to plot
    image_to_plot = glob.glob(args.test + '*.jpg')

    # plot images
    figure, axes = plt.subplots(len(image_to_read), 2, figsize=(15, 35))
    figure.tight_layout()
    flag = True
    for i, frame in enumerate(image_to_read): #    range(len(images_titles)):
        # read in an image:
        image = mpimg.imread( frame )
        # create a warped binary image, a.k.a bird's eye view
        warped = birds_eye_view(args, image)
        # set up the orverall class to do all the tracking # (50, 80) (25, 80) # (25, 15) (100, 15) (30, 20) (100, 20) # (1, 1) (3.7/783, 10/720) 
        (window_width, window_height) = (25, 80)  # the window pixel (width, height) of the center values, used to count pixels inside center windows to determine curve values
        (margin, smooth_factor)       = (100, 10) # the pixel distance in both directions to slide (left_window + right_window) template for searching
        (xm, ym)                      = (3.7/783, 10/720) # meters per pixel in (horizontal, vertical) axis

        curve_centers = TRACKER(Mywindow_width=window_width, Mywindow_height=window_height,
                                Mymargin=margin, My_ym=ym_per_pix, My_xm=xm_per_pix,
                                Mysmooth_factor=smooth_factor)

        # slide Windows, and fit a Polynomial
        output = search_sliding_window(args, warped, curve_centers)
        # plot images
        axes[i, 0].imshow(image, cmap='gray')  # display the input image
        axes[i, 0].set_title('Input '+str(i), fontsize=15)
        axes[i, 1].imshow(output, cmap='gray') # display the final result
        axes[i, 1].set_title('Output '+str(i), fontsize=15)
    plt.xlim([0, 1280])
    plt.ylim([0, 720])
    plt.show()

if __name__ == '__main__':
    main()