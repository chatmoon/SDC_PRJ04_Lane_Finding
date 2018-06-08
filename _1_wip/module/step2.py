from step0 import PARSE_ARGS
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import pickle

# parameter
directory = 'D:/USER/_PROJECT_/_PRJ04_/_1_WIP/_1_forge/_5_v4/'
args = PARSE_ARGS(path=directory)

# Helper function:
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def sobel_xy(gray, sobel_kernel=3, absolute=True):
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    if absolute:
        sobel_x, sobel_y = np.absolute(sobel_x), np.absolute(sobel_y)
    return sobel_x, sobel_y

def sobel_abs(gray, sobel_kernel=3, orient='x'):
    orientation = {'x': 0,'y': 1}[orient]
    sobel       = sobel_xy(gray, sobel_kernel=sobel_kernel, absolute=True)
    return sobel[orientation]

def scale(sobel, MAX=255, dtype=np.uint8):
    return dtype(MAX * sobel / np.max(sobel))

def mask(image, thresh=(0, 255)):
    return (image >= thresh[0]) & (image <= thresh[1])

def gradient_sobel_abs(image, sobel_kernel=3, orient='x', thresh=(0, 255), to_gray=True):
    if to_gray:
        gray = grayscale(image)
    else:
        gray = image
    sobel = sobel_abs(gray, sobel_kernel=sobel_kernel, orient=orient)
    sobel_scale = scale(sobel, MAX=thresh[1], dtype=np.uint8)
    return mask(sobel_scale, thresh)

def gradient_magnitude(image, sobel_kernel=3, thresh=(0, 255), to_gray=True):
    if to_gray:
        gray = grayscale(image)
    else:
        gray = image
    gradmag = np.hypot(*sobel_xy(gray, sobel_kernel=sobel_kernel, absolute=False))
    gradmag = scale(gradmag, MAX=thresh[1], dtype=np.uint8)
    return mask(gradmag, thresh)

def gradient_direction(image, sobel_kernel=15, thresh=(0, np.pi/2), to_gray=True):
    if to_gray:
        gray = grayscale(image)
    else:
        gray = image
    absgraddir = np.arctan2(*sobel_xy(gray, sobel_kernel=sobel_kernel, absolute=True))
    return mask(absgraddir, thresh)

# Helper function: apply threshold on grayscale image
def threshold_color_gray(image, thresh=(180, 255), binary=True):
    gray = grayscale(image)
    if binary:
        return mask(gray, thresh)
    return gray

# Helper function: apply threshold on RGB image
def threshold_color_rgb(image, color='r', thresh=(200, 255), binary=True):
    index   = {'r': 0, 'g': 1, 'b': 2}[color]
    channel = image[:, :, index]
    if binary:
        return mask(channel, thresh)
    return channel

# Helper function: apply threshold on HLS image
def threshold_color_hls(image, color='h', thresh=(90, 255), binary=True):
    index   = {'h': 0, 'l': 1, 's': 2}[color]
    hls     = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    channel = hls[:, :, index]
    if binary:
        return mask(channel, thresh)
    return channel

# Helper function: apply threshold on HLS image
def threshold_color_hsv(image, color='h', thresh=(50, 255), binary=True):
    index   = {'h': 0, 's': 1, 'v': 2}[color]
    hsv     = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    channel = hsv[:, :, index]
    if binary:
        return mask(channel, thresh)
    return channel

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def combine_threshold(args, image, stacked_image = False, region = True ): # version n-1 : _180518-1002_step3.py
    # combine several possible thresholds
    dict_binary = {'gx': (gradient_sobel_abs(image, orient='x', sobel_kernel=args.ksize, thresh=(20, 100)) == 1),
                   'gy': (gradient_sobel_abs(image, orient='y', sobel_kernel=args.ksize, thresh=(20, 100)) == 1),
                   'mb': (gradient_magnitude(image, sobel_kernel=args.ksize, thresh=(20, 100)) == 1),
                   'db': (gradient_direction(image, sobel_kernel=15, thresh=(0.7, 1.3)) == 1),
                   'y': (threshold_color_gray(image, thresh=(180, 255)) == 1),
                   'r': (threshold_color_rgb(image, color='r', thresh=(200, 255)) == 1),
                   'g': (threshold_color_rgb(image, color='g', thresh=(200, 255)) == 1),
                   'b': (threshold_color_rgb(image, color='b', thresh=(200, 255)) == 1),
                   'h': (threshold_color_hls(image, color='h', thresh=(15, 100)) == 1),
                   'l': (threshold_color_hls(image, color='l', thresh=(240, 255)) == 1),
                   's': (threshold_color_hls(image, color='s', thresh=(100, 255)) == 1),
                   'v': (threshold_color_hsv(image, color='v', thresh=(50, 255)) == 1)}
    mask1, mask4, result = [np.zeros_like(image[:, :, 0]) for i in range(3)]
    mask1[ ( dict_binary['gx'] & dict_binary['gy'] ) ] = 1
    # mask2[ ( dict_binary['mb'] & dict_binary['db'] ) ] = 1
    # mask3[ ( dict_binary['r']  & dict_binary['s']  ) ] = 1
    mask4[ ( dict_binary['s']  & dict_binary['v']  ) ] = 1
    # mask5[ ( dict_binary['r']  & dict_binary['s'] & dict_binary['v']  ) ] = 1

    #result[ ( dict_binary['gx'] & dict_binary['gy'] ) |  ( dict_binary['s']  & dict_binary['v']  ) ] = 1
    #result[(dict_binary['gx'] & dict_binary['gy']) | ( dict_binary['mb'] & dict_binary['db'] ) | (dict_binary['r'] & dict_binary['s'] & dict_binary['v'])] = 1
    result[(dict_binary['gx'] & dict_binary['gy']) | (dict_binary['r']  & dict_binary['s'] & dict_binary['v'])] = 255 # 1

    if region:
        vertices = np.array([ (570, 449), (700, 449), (1200, 665), (62, 665) ])
        vertices = np.int32([vertices])
        result   = region_of_interest(result, vertices)

    if stacked_image:
        return np.dstack((np.zeros_like( mask1 ) , mask1, mask4 )) * 255
    else:
        return result


def main():
    # parameter
    directory = 'D:/USER/_PROJECT_/_PRJ04_/_1_WIP/_1_forge/_3_retro/'
    args = PARSE_ARGS(path=directory)
    # choose a Sobel kernel size
    ksize = 3
    # read images
    image        = mpimg.imread(args.sand+'signs_vehicles_xygrad.jpg')
    img_solution = mpimg.imread(args.sand+'binary-combo-example.jpg')
    # apply each of the thresholding functions
    gradx = gradient_sobel_abs(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = gradient_sobel_abs(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = gradient_magnitude(image, sobel_kernel=ksize, thresh=(20, 100))
    dir_binary = gradient_direction(image, sobel_kernel=15, thresh=(0.7, 1.3))
    
    # combine thresholds
    combined1, combined2, combined3, combined4, combined5, combined6, combined7 = [np.zeros_like(dir_binary) for i in range(7)]
    combined1[((gradx == 1) & (grady == 1))] = 1
    combined2[((mag_binary == 1) & (dir_binary == 1))] = 1
    combined3[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined4[((gradx == 1) & (grady == 1) & (mag_binary == 1))] = 1
    combined5[((gradx == 1) & (grady == 1) | (mag_binary == 1))] = 1
    combined6[((gradx == 1) & (grady == 1) & (dir_binary == 1))] = 1
    combined7[((gradx == 1) & (grady == 1) | (dir_binary == 1))] = 1

    # plot the result
    row, column = [12, 2]
    figure, axes = plt.subplots(row, column, figsize=(15, 50))
    figure.tight_layout()
    expected_result  = ['Expected result', img_solution]
    list_title_image = [['Original Image',image],
                        ['gradx', gradx],
                        ['grady', grady],
                        ['mag_binary', mag_binary],
                        ['dir_binary', dir_binary],                        
                        ['comb1: gradx & grady', combined1],
                        ['comb2: mag_binary & dir_binary', combined2],
                        ['comb3: <gradx & grady> OR <dir_bin & mag_bin> ', combined3],
                        ['comb4: gradx & grady & mag_binary', combined4],
                        ['comb5: gradx & grady | mag_binary', combined5],
                        ['comb6: gradx & grady & dir_binary', combined6],
                        ['comb7: gradx & grady | dir_binary', combined7] ]

    count = 0
    for i, ax in enumerate(axes.flatten()):
        if i%2==0:            
            ax.imshow(expected_result[1], cmap='gray')
            ax.set_title(expected_result[0], fontsize=15)
        else:
            ax.imshow(list_title_image[count][1], cmap='gray')
            ax.set_title(list_title_image[count][0], fontsize=15)    
            count += 1 
        ax.axis('off')           
   
if __name__ == '__main__':
    main()