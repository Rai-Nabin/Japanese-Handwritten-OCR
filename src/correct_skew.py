import cv2 as cv
import numpy as np


def show_image(image, window_name):

    cv.imshow(window_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_image_size(image):

    height = image.shape[0]
    width = image.shape[1]

    return (height, width)


def detect_corners(image):
    corners = cv.goodFeaturesToTrack(image, maxCorners=4, qualityLevel=0.1, minDistance=30)
    return corners


def sort_corners(original_image, corners):
    rectangular_box = cv.minAreaRect(corners)

    angle_of_inclination = rectangular_box[-1]
    coordinates = cv.boxPoints(rectangular_box)

    threshold = -10.0
    if angle_of_inclination > threshold:
        # Left shift by one
        coordinates = np.roll(coordinates, 2)

    arranged_incdices = [2, 3, 1, 0]
    coordinates = [coordinates[i] for i in arranged_incdices]

    # Visualize points
    # for i in coordinates:
    #     cv.circle(original_image, (i[0], i[1]), 4, (0, 255, 0), -1)
    #     show_image(original_image, "Gray Image")

    return coordinates


def get_destination_corners(four_corners):

    # Flatten nested numpy list
    four_corners = np.concatenate(four_corners).tolist()

    x1, y1 = four_corners[:2]
    x2, y2 = four_corners[2:4]
    x3, y3 = four_corners[6:8]
    x4, y4 = four_corners[4:6]

    # Compute the width of the new image, which will be the maximum distance between x-coordinates
    w1 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    w2 = np.sqrt((x3-x4)**2 + (y3-y4)**2)

    max_width = max(int(w1), int(w2))

    # Compute the height of the new image, which will be the maximum distance between y-coordinates
    h1 = np.sqrt((x4-x1)**2 + (y4-y1)**2)
    h2 = np.sqrt((x3-x2)**2 + (y3-y2)**2)

    max_height = max(int(h1), int(h2))

    destination_corners = np.array(
        [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]], dtype="float32")

    return destination_corners, max_width, max_height


def unwarp_image(image, four_corners, destination_corners):

    height, width = get_image_size(image)

    # Convert four corners to required format
    four_corners = np.concatenate(four_corners).tolist()
    point1 = []
    length = len(four_corners)
    i = 0
    while i < length:
        point1.append([four_corners[i], four_corners[i+1]])
        i += 2
    
    point1 = np.float32(point1)
    point2 = destination_corners

    H, _ = cv.findHomography(
        point1, point2, method=cv.RANSAC, ransacReprojThreshold=3.0)

    unwarpped_image = cv.warpPerspective(
        image, H, (width, height), flags=cv.INTER_LINEAR)

    # show_image(unwarpped_image, "Final Image")
    return unwarpped_image


def crop_image(image, max_width, max_height):
    # To add padding(bottom and right side of image)
    offset = 4
    
    origin = 5
    deskewed_cropped_image = image[origin:max_height-offset, origin:max_width-offset]

    # show_image(deskewed_cropped_image, "Cropped_image")

    return deskewed_cropped_image
    
def apply_filter(gray_image):
    kernel = np.ones((11, 11), np.float32)/10
    filtered_2D_image = cv.filter2D(gray_image, ddepth=-1, kernel=kernel)
    # show_image(filtered_2D_image, "Filtered Image")
    return filtered_2D_image

def correct_skew(cropped_image):

   
    gray_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)
    # show_image(gray_image, "Gray Image")
    filtered_2D_image = apply_filter(gray_image)
    
    corners = detect_corners(filtered_2D_image)
    
    sorted_corners = sort_corners(filtered_2D_image, corners)

    destination_corners, max_width, max_height = get_destination_corners(
        sorted_corners)

    unwarpped_image = unwarp_image(
        cropped_image, sorted_corners, destination_corners)
    # show_image(unwarpped_image, "Demo")

    deskewed_image = crop_image(unwarpped_image, max_width, max_height)

    return deskewed_image


