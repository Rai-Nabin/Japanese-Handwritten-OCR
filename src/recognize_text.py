import numpy as np
from itertools import groupby
from openvino.inference_engine import IECore

class TextRecognition:
    def __init__(self, model_path):
        self.model_path = model_path
        self.network, self.executable_network = self.load_to_IE()
        self.input_layer, self.output_layer = self.get_model_layers()
        self.H, self.W = self.get_model_layer_info()

    def load_to_IE(self):
        # Load the Inference Engine API
        ie = IECore()
        # Load the IR files
        network = ie.read_network(model=f'{self.model_path}.xml', weights=f'{self.model_path}.bin')
        executable_network = ie.load_network(network=network, device_name="CPU")
        return network, executable_network

    def get_model_layers(self):
        input_layer = next(iter(self.network.input_info))
        output_layer = next(iter(self.network.outputs))
        return input_layer, output_layer

    def get_model_layer_info(self):
        _, _, H, W = self.network.input_info[self.input_layer].input_data.shape
        return H, W

    def get_predictions_index(self, input_image):
        predictions = self.executable_network.infer(inputs={self.input_layer: input_image})[self.output_layer]
        predictions = np.squeeze(predictions)
        predictions_index = np.argmax(predictions, axis=1)
        return predictions_index

    @staticmethod
    def prepare_charlist():
        blank_char = "~"
        with open('charlists/japanese_charlist.txt', "r", encoding="utf-8") as charlist:
            letters = blank_char + "".join(line.strip() for line in charlist)
        return letters

    @staticmethod
    def get_text_from_predictions_index(predictions_index, letters):
        output_text_indexes = list(groupby(predictions_index))
        output_text_indexes, _ = np.transpose(output_text_indexes)
        output_text_indexes = output_text_indexes[output_text_indexes != 0]
        output_char = [letters[letter_index] for letter_index in output_text_indexes]
        output_text = "".join(output_char)
        return output_text
