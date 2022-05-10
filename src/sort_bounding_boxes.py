
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
