import sys
from itertools import groupby
import os
import shutil
import cv2 as cv
import numpy as np
from openvino.inference_engine import IECore

from src.crop_image import crop_image
from src.detect_text import (detect_text_cooridinates,
                             draw_bounding_box_and_save)
from src.pre_process import apply_threshold, resize_image, show_image
from src.save_output import save_to_json
from src.sort_bounding_boxes import sort_bounding_boxes


def load_to_IE(path_to_model):
    # Loading the Inference Engine API
    ie = IECore()

    # Loading IR files
    network = ie.read_network(
        model=f'{path_to_model}.xml', weights=f'{path_to_model}.bin')

    executable_network = ie.load_network(
        network=network, device_name="GPU")

    return network, executable_network


def get_model_layer(network):

    input_layer = next(iter(network.input_info))
    output_layer = next(iter(network.outputs))

    return input_layer, output_layer


def get_model_layer_info(network, input_layer):
    # B, C, H, W = batch size, number of channels, height, width
    _, _, H, W = network.input_info[input_layer].input_data.shape

    return H, W


def prepare_charlist():
    blank_char = "~"
    with open(f'charlists/japanese_charlist.txt', "r", encoding="utf-8") as charlist:
        letters = blank_char + "".join(line.strip() for line in charlist)
    return letters


def get_predictions_index(input_image, executable_network, input_layer, output_layer):
    # Run inference on the model
    predictions = executable_network.infer(
        inputs={input_layer: input_image})[output_layer]

    # Remove batch dimension
    predictions = np.squeeze(predictions)
    # Run argmax to pick the symbols with the highest probability
    predictions_index = np.argmax(predictions, axis=1)

    return predictions_index


def get_text_from_predictions_index(predictions_index, letters):
    # Use groupby to remove concurrent letters, as required by CTC greedy decoding
    output_text_indexes = list(groupby(predictions_index))

    # Remove grouper objects
    output_text_indexes, _ = np.transpose(output_text_indexes)

    # Remove blank symbols
    output_text_indexes = output_text_indexes[output_text_indexes != 0]

    # Assign letters to indexes from output array
    output_char = [letters[letter_index]
                   for letter_index in output_text_indexes]

    output_text = "".join(output_char)
    return output_text


def clean_folder(folder_name):
    folder_path = f'images/{folder_name}'

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    else:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)


def main():
    INPUT_IMAGE_PATH = sys.argv[1]
    FILE_NAME = INPUT_IMAGE_PATH.split('/')[-1].split('.')[0]

    network, executable_network = load_to_IE(
        'model/handwritten-japanese-recognition-0001/FP16/handwritten-japanese-recognition-0001')
    input_layer, output_layer = get_model_layer(network)

    H, W = get_model_layer_info(network, input_layer)

    # Read image and convert it into RGB format
    image = cv.imread(INPUT_IMAGE_PATH)
    # show_image(image, "Input Image")

    # Compute height and width of image
    image_height = image.shape[0]
    image_width = image.shape[1]

    bounding_boxes = detect_text_cooridinates(INPUT_IMAGE_PATH)
    sorted_bounding_boxes = sort_bounding_boxes(bounding_boxes)

    crop_image(image, sorted_bounding_boxes, image_height, image_width)

    draw_bounding_box_and_save(image, sorted_bounding_boxes)

    letters = prepare_charlist()
    extracted_text = {}

    clean_folder(folder_name='pre_processed_image')
    clean_folder(folder_name='padded_image')

    for idx in range(len(sorted_bounding_boxes)):

        apply_threshold(idx)
        input_image = resize_image(H, W, idx)

        predictions_index = get_predictions_index(input_image, executable_network,
                                                  input_layer, output_layer)
        output_text = get_text_from_predictions_index(
            predictions_index, letters)

        extracted_text[idx+1] = output_text
    save_to_json(extracted_text)


if __name__ == "__main__":
    main()
