import cv2 as cv
import numpy as np

class CropImage:

    @staticmethod
    def sort_bounding_boxes(bounding_boxes):
        """Takes first point from each bounding boxes. Then it sorts y-coordinates and if two points have same y-cordinates it looks for x-coordinates.

        Args:
            bounding_boxes (list): Bounding boxes coordinates of text.

        Returns:
            list: [[[544.0, 261.0], [1935.0, 226.0], [1938.0, 352.0], [548.0, 387.0]], ...]
        """
        sorted_bounding_boxes = sorted(
            bounding_boxes, key=lambda k: (k[:][0][1], k[:][0][0]))

        return sorted_bounding_boxes

    @staticmethod
    def crop_image(image, bounding_boxes, path_to_save):
        """Crops bounding boxes from the image.

        Args:
            image (uint8): Input image
            bounding_boxes (list): Bounding boxes of text
        """
        image_height, image_width = image.shape[:2]

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
            offset = 10

            cropped_image = masked_image[rectangle[1]-offset: rectangle[1] +
                                        rectangle[3]+offset, rectangle[0]-offset: rectangle[0] + rectangle[2] + offset]

            cv.imwrite(f'{path_to_save}/crop_{idx}.jpg', cropped_image)

    @staticmethod
    def sort_bounding_boxes_and_crop_image(image, bounding_boxes, path_to_save):
        sorted_bounding_boxes = CropImage.sort_bounding_boxes(bounding_boxes)
        CropImage.crop_image(image, sorted_bounding_boxes, path_to_save)


            