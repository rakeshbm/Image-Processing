import random
import math
import copy
from image_builder import Builder

class RandomErasingBuilder(Builder):
    def __init__(self, params):
        Builder.__init__(self, params, builder_name='random_erasing')

    def _core_augment_algorithm(self):
        """Implements copy pasting algorithm"""

        input_img_data = self._img_data.input_data
        new_img = copy.deepcopy(input_img_data)

        width = input_img_data.shape[1]
        height = input_img_data.shape[0]
        area = width * height

        for i in range(100):
            aspect_ratio = random.uniform(0.3, 1 / 0.3)
            target_area = random.uniform(0.05, 0.2) * area
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if h < height and w < width:
                x1 = random.randint(0, width - w)
                y1 = random.randint(0, height - h)
                new_img[y1:y1 + h, x1:x1 + w] = 0
                return new_img
        return new_img
