from step0 import PARSE_ARGS
from step1 import image_undistort
from step2 import combine_threshold
# grayscale, sobel_xy, sobel_abs, scale, mask, gradient_sobel_abs, gradient_magnitude, gradient_direction, threshold_color_gray, threshold_color_rgb, threshold_color_hls, threshold_color_hsv, region_of_interest, combine_threshold

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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

def birds_eye_view(args, image, region = False):
    # undistort and transform perspective
    image_undistorted = image_undistort(args, image)
    # combine several possible thresholds
    image_binary = combine_threshold(args, image_undistorted, stacked_image = False, region = region )
    # transform perspective
    _, _, warped = image_warp(args, image_binary, x_top = [ 594, 686 ])
    return warped


def test_birds_eye_view1(args, image):
    '''
    test of different source coordinates
    '''
    # undistort using mtx and dist
    image_undistorted = image_undistort(args, image)
    # undistort, transform perspective, plot images
    lower, upper = ( 592, 684 ) , ( 592, 692 )
    j = 6 # 10
    mag = int( ( upper[1] - lower[1] )/j )
    images_titles = []

    for i in range(j):
        _, _, warped = image_warp(args, image_undistorted, x_top = [lower[0]+mag*i, lower[1]+mag*i])
        images_titles.append(['< '+str(lower[0]+mag*i)+', '+str(lower[1]+mag*i)+' >', warped ])

    image_white = np.zeros_like(images_titles[0][1])
    image_white.fill(255)

    column = 2
    figure, axes = plt.subplots(len(images_titles)//column, column, figsize=args.figsize)
    figure.tight_layout()

    for i, ax in enumerate(axes.flatten()):
        if i < len(images_titles):
            ax.imshow(images_titles[i][1], cmap='gray')
            ax.set_title(images_titles[i][0], fontsize=15)
        else:
            ax.imshow(image_white)
        #ax.axis('off')
        ax.grid(color='w', linestyle='--', linewidth=1)
    plt.show()


def test_birds_eye_view2(args, image_to_read, region = False):
    # define the zone of interest with four points coordinates
    #points = np.array([[594, 449], [686, 449], [1045, 665], [262, 665]], np.int32)
    points = np.array([[570, 449], [750, 449], [1150, 665], [262, 665]], np.int32)
    # undistort and transform perspective
    images, warped = [], []
    for frame in image_to_read:
        # read in an image
        image = mpimg.imread(frame)
        # transform perspective
        warped.append(birds_eye_view(args, image, region = region))
        # plot the region of interest
        images.append(cv2.polylines(image_undistort(args, image), [points], True, color=(0, 0, 0), thickness=8))

    # plot images
    figure, axes = plt.subplots(len(image_to_read), 2, figsize=args.figsize)  # (15, 10)
    figure.tight_layout()
    for i, (image, image_warped) in enumerate(zip(images, warped)):  # range(len(images_titles)):
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title('Input ' + str(i), fontsize=15)
        axes[i, 1].imshow(image_warped, cmap='gray')
        axes[i, 1].set_title('Bird\'s eye-view ' + str(i), fontsize=15)
        axes[i, 1].grid(color='w', linestyle='--', linewidth=1)
    plt.show()


def histograms_plot(args, image_to_plot, region = False):
    # plot images
    figure, axes = plt.subplots(len(image_to_plot), 3, figsize=(15, 25))
    figure.tight_layout()
    for i, frame in enumerate(image_to_plot):
        # read in an image:
        image = mpimg.imread(frame)
        # create a warped binary image, a.k.a bird's eye view
        image_binary_warped = birds_eye_view(args, image, region = region)
        # find lane lines with histogram
        histogram = np.sum(image_binary_warped[image_binary_warped.shape[0] // 2:, :], axis=0)
        # plot images
        axes[i, 0].imshow(image, cmap='gray')  # display the input image
        axes[i, 0].set_title('input ' + str(i), fontsize=15)
        axes[i, 1].imshow(image_binary_warped, cmap='gray')  # display the warped binary image
        axes[i, 1].set_title('binary image ' + str(i), fontsize=15)
        axes[i, 2].plot(histogram, color='black')  # display the final result
        axes[i, 2].set_xlabel('pixel position')
        axes[i, 2].set_ylabel('counts')
        axes[i, 2].set_title('histogram ' + str(i), fontsize=15)
    plt.show()



def main():
    # parameter
    directory = 'D:/USER/_PROJECT_/_PRJ04_/_1_WIP/_1_forge/_5_v4/'
    args = PARSE_ARGS(path=directory)
    # read in an image:
    image_to_read = [args.test + 'straight_lines1.jpg', args.test + 'straight_lines2.jpg']
    image = mpimg.imread(image_to_read[0])
    # test of different source coordinates
    test_birds_eye_view1(args, image)
    # test of different image input
    test_birds_eye_view2(args, image_to_read)

if __name__ == '__main__':
    main()