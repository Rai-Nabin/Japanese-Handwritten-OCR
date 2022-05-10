import cv2 as cv
import numpy as np
import os
import shutil
from src.correct_skew import correct_skew
from src.pre_process import show_image


def crop_image(image, bounding_boxes, image_height, image_width):
    """Crops bounding boxes from the image.

    Args:
        image (uint8): Input image
        bounding_boxes (list): Bounding boxes of text
        image_height (int): Height of input image
        image_width (int): Width of input image
    """
    folder_path = f'images/cropped_image'
    
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    else:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)

    for idx, coordinates in enumerate(bounding_boxes):

        points = np.array([coordinates], dtype=np.int32)

        # Returns lowest (x-axis, y-axis), maximumum(width and height) of the rectangle
        rectangle = cv.boundingRect(points)

        # Create mask of the same size of image
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        cv.fillPoly(mask, [points], (255, 255, 255))

        # Apply mask to the original image
        masked_image = cv.bitwise_and(image, image, mask=mask)
        
        
        # To add padding to the cropped image
        offset = 0

        cropped_image = masked_image[rectangle[1]-offset: rectangle[1] +
                                     rectangle[3]+offset, rectangle[0]-offset: rectangle[0] + rectangle[2] + offset]

        # show_image(cropped_image, "Cropped Image")
       
        deskewed_image = correct_skew(cropped_image)

        cv.imwrite(f'images/cropped_image/crop_{idx}.jpg', deskewed_image)
        # cv.imwrite(f'images/cropped_image/crop_{idx}.jpg', cropped_image)
        
