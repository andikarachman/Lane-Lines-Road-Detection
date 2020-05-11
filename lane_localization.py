import argparse
import glob
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob

import camera_calibration
import perspective_transform
import thresholding


def get_binary_warped(image_file):
    # read in image
    img = mpimg.imread(image_file)

    # undistort image
    params = camera_calibration.load_calibration_data()
    img = camera_calibration.undistort_image(img, params)

    # perform perspective transform
    img, M, Minv = perspective_transform.apply_transform(img)

    # perform thresholding
    processed_image = thresholding.process_imaging(img)

    return img, processed_image


def histogram(binary_warped):
    # Take a histogram of the bottom half of 
    # the binary warped image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] * 0.5):,:], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

     # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return out_img, left_fit, right_fit


def find_lane_lines(binary_warped, left_fit, right_fit):
    # the +/- margin of our polynomial function
    margin = 25
    
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the area of search based on activated x-values
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + \
                       left_fit[1] * nonzeroy + left_fit[2] - margin)) &
                      (nonzerox < (left_fit[0] * (nonzeroy ** 2) + \
                       left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + \
                        right_fit[1] * nonzeroy + right_fit[2] - margin)) &
                       (nonzerox < (right_fit[0] * (nonzeroy ** 2) + \
                        right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting and calculate curvature
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Calculate curvature
    left_curverad, right_curverad = calculate_curvature(ploty, left_fitx, right_fitx)

    # Calculate car position
    car_pos = calculate_car_position(binary_warped, ploty, left_fit, right_fit)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.transpose(np.vstack([left_fitx + margin, ploty]))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane lines onto the warped blank image
    cv2.fillPoly(out_img, np.int_([left_line_pts]), (255, 0, 0))
    cv2.fillPoly(out_img, np.int_([right_line_pts]), (0, 0, 255))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(out_img, np.int_([pts]), (0, 255, 0))

    return out_img, left_curverad, right_curverad, car_pos


def calculate_curvature(ploty, left_fitx, right_fitx):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit a second order polynomial to pixel positions in each fake lane line
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad

def calculate_car_position(img, ploty, left_fit, right_fit):
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # find x values of leftftix and rightfitx at the bottom of the image (y max) and 
    y_eval = np.max(ploty)
    x_left_eval = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    x_right_eval = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    
    # find the mid point between left and right lane
    car_location = x_left_eval + (x_right_eval - x_left_eval) / 2
    car_offset = img.shape[1] / 2 - car_location
    car_offset_meter = car_offset * xm_per_pix

    return car_offset_meter


def show_draw_lines(image_file, visualize=False, save_example=False):
    image, binary_warped = get_binary_warped(image_file)
    histogram_image, left_fit, right_fit = histogram(binary_warped)
    result, _, _, _ = find_lane_lines(binary_warped, left_fit, right_fit)
    
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=24)
    ax2.imshow(result, cmap="gray")
    ax2.set_title("Lane Lines Result", fontsize=24)
    if visualize:
        plt.show(block=True)
    if save_example:
        save_file_name = "lane_lines_{}".format(os.path.basename(image_file.replace(".jpg", ".png")))
        save_location = "./output_images/{}".format(save_file_name)
        f.savefig(save_location, bbox_inches="tight")


def show_histogram(image_file, visualize=False, save_example=False):
    image, binary_warped = get_binary_warped(image_file)
    result, left_fit, right_fit = histogram(binary_warped)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=24)
    ax2.imshow(result, cmap="gray")
    ax2.set_title("Histogram Result", fontsize=24)
    if visualize:
        ax2.plot(left_fitx, ploty, color='yellow')
        ax2.plot(right_fitx, ploty, color='yellow')
        plt.show(block=True)
    if save_example:
        save_file_name = "histogram_{}".format(os.path.basename(image_file.replace(".jpg", ".png")))
        save_location = "./output_images/{}".format(save_file_name)
        f.savefig(save_location, bbox_inches="tight")

def put_text(img, left_curverad, right_curverad, car_pos):
    # Write the radius of curvature
    left = "Left Radius of Curvature = {0: >3.0f}m".format(left_curverad)
    right = "Right Radius of Curvature = {0: >3.0f}m".format(right_curverad)
    if car_pos <= 0 :
        direction = 'right of center'
    else:
        direction = 'left of center'
    car_offset_text = 'Car offset: '+ '{:04.3f}'.format(abs(car_pos)) +' m ' + direction

    cv2.putText(img, left, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, right, (50, 85), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, car_offset_text, (50, 120), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Histogram and lane line fitting")
    parser.add_argument("-show", action="store_true", help="Visualize histogram and lane line images")
    parser.add_argument("-save", action="store_true", help="Save histogram and lane line image")
    results = parser.parse_args()
    visualize = bool(results.show)
    save_examples = bool(results.save)

    images = glob.glob("./test_images/test*.jpg")
    for file_name in images:
        show_histogram(file_name, visualize, save_examples)
        show_draw_lines(file_name, visualize, save_examples)




    
    








