import cv2 as cv
import os
import sys
import shutil
from src.detect_text import TextDetection
from src.pre_process import PreProcessing
from src.crop_image import CropImage
from src.recognize_text import TextRecognition
from src.save_output import save_to_json


def show_image(image, window_name):
    cv.imshow(window_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def make_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    
def del_folder(folder_path):
    shutil.rmtree(folder_path)

def list_file_paths_in_folder(folder_path):
    file_paths = []
    try:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            # Check if it is a file
            if os.path.isfile(file_path):
                file_paths.append(file_path)
        # Sort the file paths in alphabetical order
        file_paths.sort()
    except Exception as e:
        print(f"An error occured: {e}")
    
    return file_paths

def main():
    INPUT_IMAGE_PATH = sys.argv[1]
    FILE_NAME = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0]

    text_detection = TextDetection(det_model_dir='model/det_model/ch_ppocr_server_v2.0_det_infer/')
    text_recognition = TextRecognition('model/handwritten-japanese-recognition-0001/FP16/handwritten-japanese-recognition-0001')

    image = cv.imread(INPUT_IMAGE_PATH)
    # show_image(image, "Input Image")

    bounding_boxes = text_detection.detect_text_coordinates(INPUT_IMAGE_PATH)
    # print("Bounding Boxes: ", bounding_boxes)
    sorted_bounding_boxes = text_detection.sort_bounding_boxes(bounding_boxes)
    # print("Sorted Bounding Boxes: ", sorted_bounding_boxes)

    # Make a temporary folder to store pre-processed images
    temp_folder = os.path.splitext(INPUT_IMAGE_PATH)[0]

    # Make a folder to store cropped images
    cropped_images_folder = f"{temp_folder}/cropped_images"
    make_folder(cropped_images_folder)

    CropImage.crop_image(image=image, bounding_boxes=sorted_bounding_boxes, path_to_save=cropped_images_folder)
    
    # Store text detected images to 'output' folder
    output_path = "output"
    make_folder(output_path)   
    text_detection.draw_bounding_box_and_save(image, sorted_bounding_boxes, output_image_path=f'{output_path}/{FILE_NAME}.jpg')

    # Make a folder to store thresholded images
    thresholded_images_folder = f"{temp_folder}/thresholded_images"
    make_folder(thresholded_images_folder)

    letters = text_recognition.prepare_charlist()
    extracted_text = {}

    cropped_image_paths = list_file_paths_in_folder(cropped_images_folder)
    for idx, cropped_image_path in enumerate(cropped_image_paths):
        PreProcessing.apply_threshold(cropped_image_path, save_path=f'{thresholded_images_folder}/threshold_{idx+1}.jpg')
        input_image = PreProcessing.resize_image(text_recognition.H, text_recognition.W, f'{thresholded_images_folder}/threshold_{idx+1}.jpg')
        predictions_index = text_recognition.get_predictions_index(input_image)
        output_text = text_recognition.get_text_from_predictions_index(predictions_index, letters)
        extracted_text[idx + 1] = output_text
    save_to_json(extracted_text, save_path=f'{output_path}/{FILE_NAME}.json')

    # Delete temporary folder created earlier
    del_folder(temp_folder)

if __name__ == "__main__":
    main()
