import argparse
import glob
import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import camera_calibration
import perspective_transform


def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    """
    Applies Sobel x or y, then takes an absolute 
    value and applies a threshold.
    """
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply x or y gradient with the OpenCV Sobel() function and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))

    # Rescale to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Define a function that applies Sobel x and y, 
    then computes the magnitude of the gradient
    and applies a threshold
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    
    # Rescale to 8 bit integer
    gradmag = np.uint8(255 * gradmag / np.max(gradmag))
    
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    # Define a function that applies Sobel x and y, 
    # then computes the direction of the gradient
    # and applies a threshold.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def color_threshold(img, hls_thresh=(0, 255), hsv_thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= hls_thresh[0]) & (s_channel <= hls_thresh[1])] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= hsv_thresh[0]) & (v_channel <= hsv_thresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1
    return output


def gaussian_blur(img, kernel=5):
    blur = cv2.GaussianBlur(img, (kernel, kernel), 0)
    return blur


def process_imaging(img):
    """
    Run the sobel and color threshold functions with bitwise mapping, 
    then returns gaussian blur of the result
    """
    sobel_x = abs_sobel_thresh(img, orient='x', thresh=(35, 255))
    sobel_y = abs_sobel_thresh(img, orient='y', thresh=(15, 255))
    sobel_binary = cv2.bitwise_and(sobel_x, sobel_y)

    color_binary = color_threshold(img, hls_thresh=(150, 255), hsv_thresh=(200, 255))

    processed_image = cv2.bitwise_or(sobel_binary, color_binary)
    
    processed_image = gaussian_blur(processed_image, kernel=9)
    
    return processed_image


def show_imaging(image_file, visualize=False, save_example=False):
    image = mpimg.imread(image_file)

    # Undistort image
    params = camera_calibration.load_calibration_data()
    image = camera_calibration.undistort_image(image, params)

    # Perspective transform
    image, M, Minv = perspective_transform.apply_transform(image)

    # Apply sobel and color thresholding
    processed_image = process_imaging(image)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=24)
    ax2.imshow(processed_image, cmap="gray")
    ax2.set_title("Processed Image", fontsize=24)
    if visualize:
        plt.show(block=True)
    if save_example:
        save_file_name = "imaging_{}".format(os.path.basename(image_file.replace(".jpg", ".png")))
        save_location = "./output_images/{}".format(save_file_name)
        f.savefig(save_location, bbox_inches="tight")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Thresholding methods for Lane Finding")
    parser.add_argument("-show", action="store_true", help="Visualize imaging process")
    parser.add_argument("-save", action="store_true", help="Save processed image")
    results = parser.parse_args()
    visualize = bool(results.show)
    save_examples = bool(results.save)
    images = glob.glob("./test_images/test*.jpg")
    for file_name in images:
        show_imaging(file_name, visualize, save_examples)