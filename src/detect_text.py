from paddleocr import PaddleOCR
import numpy as np
import cv2 as cv

class TextDetection:
    def __init__(self, det_model_dir):
        self.ocr = PaddleOCR(det_model_dir=det_model_dir, use_gpu=False)

    def detect_text_coordinates(self, image_path):
        """Detects bounding boxes coordinates of text.

        Args:
            image (uint8): Input image for text detection.

        Returns:
            list: [[[1015.0, 2788.0], [1809.0, 2798.0], [1807.0, 2936.0], [1013.0, 2927.0]], ...]
        """
        bounding_boxes = self.ocr.ocr(image_path, rec=False, cls=False)
        return bounding_boxes
    
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

    def draw_bounding_box_and_save(self, image, bounding_boxes, output_image_path):
        """Draws bounding boxes around text image and saves it.

        Args:
            image (uint8): Input image
            bounding_boxes (list): Bounding boxes coordinates
        """

        # Converts to Format: [array([[[ 544.,  261.]],[[1935.,  226.]],[[1938.,  352.]],[[ 548.,  387.]]], dtype=float32),...]
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
            cv.putText(image, str(index + 1), (x + w // 4, y + h // 4),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 0, 0), 3, cv.LINE_AA)
            cv.imwrite(output_image_path, image)

