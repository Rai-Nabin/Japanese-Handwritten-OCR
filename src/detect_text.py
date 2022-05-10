from paddleocr import PaddleOCR
import numpy as np
import cv2 as cv
from src.pre_process import show_image


def detect_text_cooridinates(image_path):
    """Detects bounding boxes coordinates of text.

    Args:
        image (uint8): Input image for text detection.

    Returns:
        list: [[[1015.0, 2788.0], [1809.0, 2798.0], [1807.0, 2936.0], [1013.0, 2927.0]], ...]
    """
    ocr = PaddleOCR(
        det_model_dir='model/det_model/ch_ppocr_server_v2.0_det_infer/')

    bounding_boxes = ocr.ocr(image_path, rec=False, cls=False)
    return bounding_boxes


def draw_bounding_box_and_save(image, bounding_boxes):
    """Draws bounding boxes around text image and saves it.

    Args:
        image (uint8): Input image
        bounding_boxes (list): Bounding boxes coordinates
    """

    # Format: [array([[[ 544.,  261.]],[[1935.,  226.]],[[1938.,  352.]],[[ 548.,  387.]]], dtype=float32),...]
    formatted_bounding_boxes = []
    for box in bounding_boxes:
        test_box = [[point] for point in box]
        test_box = np.float32(test_box)
        formatted_bounding_boxes.append(test_box)

    for index, box in enumerate(formatted_bounding_boxes):
        x, y, w, h = cv.boundingRect(box)

        rectangular_box = cv.minAreaRect(box)
        coordinates = np.int0(cv.boxPoints(rectangular_box))
        cv.drawContours(image, [coordinates], 0, (0, 0, 255), 3, cv.LINE_AA)
        cv.putText(image, str(index+1), (x+w//4, y+h//4),
                   cv.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 0, 0), 3, cv.LINE_AA)

        cv.imwrite("output/ouput.jpg", image)
