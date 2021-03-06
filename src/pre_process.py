import os
import shutil

import cv2 as cv
import numpy as np


def show_image(image, window_name):
    cv.imshow(window_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def apply_threshold(idx):

    image = cv.imread(f'images/cropped_image/crop_{idx}.jpg', 0)

    thresholded_image = cv.threshold(
        image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    # show_image(thresholded_image, '')

    cv.imwrite(
        f'images/pre_processed_image/preprocessed_{idx}.jpg', thresholded_image)


def resize_image(H, W, idx):
    image = cv.imread(
        f'images/pre_processed_image/preprocessed_{idx}.jpg', flags=cv.IMREAD_GRAYSCALE)
    image_height, _ = image.shape

    # Calculate scale ratio between input shape height and image height to resize image
    scale_ratio_height = H/image_height

    # Reszie image to expected input sizes(adjust height)
    resized_image = cv.resize(
        image, None, fx=scale_ratio_height, fy=scale_ratio_height, interpolation=cv.INTER_AREA)

    resized_image_width = resized_image.shape[1]
    # Calculate scale ratio between width of resized image and width of model
    scale_ratio_width = W/resized_image_width

    if scale_ratio_width < 1:
        resized_image = cv.resize(resized_image, None, fx=scale_ratio_width,
                                  fy=scale_ratio_width, interpolation=cv.INTER_AREA)

        height_difference = (H-resized_image.shape[0])

        if height_difference % 2 == 0:
            pad_length = height_difference//2
            padded_image = cv.copyMakeBorder(
                resized_image, top=pad_length, bottom=pad_length, right=0, left=0, borderType=cv.BORDER_CONSTANT, value=255)
        else:
            pad_length = height_difference//2
            padded_image = cv.copyMakeBorder(
                resized_image, top=pad_length, bottom=pad_length+1, right=0, left=0, borderType=cv.BORDER_CONSTANT, value=255)
    else:

        padded_image = cv.copyMakeBorder(resized_image, top=0, bottom=0, right=(
            W-resized_image.shape[1]), left=0, borderType=cv.BORDER_CONSTANT, value=255)

    width, height = padded_image.shape
    print("Padded Image:", idx, width, height)
    cv.imwrite(f'images/padded_image/padded_{idx}.jpg', padded_image)
    # Reshape to network the input shape
    input_image = padded_image[None, None, :, :]

    return input_image
