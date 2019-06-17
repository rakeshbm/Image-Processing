import os
import cv2
import random
import copy
from image_builder import Builder


class MixupBuilder(Builder):
    def __init__(self, params):
        Builder.__init__(self, params, builder_name='mix_up')
        self._alpha = params['alpha']

    def _core_augment_algorithm(self):
        """Implements mix up algorithm"""
        img1 = self._img_data.input_data
        img2 = self._generate_mixup_img()

        resize_img2 = cv2.resize(img2,
                                (img1.shape[1], img1.shape[0]),
                                interpolation=cv2.INTER_CUBIC)
        new_img = (img1 * self._alpha + resize_img2 * (1 - self._alpha))
        return new_img

    def _generate_mixup_img(self):
        """Generates mix up img"""
        input_img_folder = self._img_data.input_folder
        filenames = os.listdir(input_img_folder)
        filenames = copy.deepcopy(filenames)
        filenames.remove(self._img_data.input_filename)
        try:
            filenames.remove(".DS_Store")
        except:
            pass
        auxiliary_img_name = filenames[random.randint(0, len(filenames)-1)]
        auxiliary_img_path = os.path.join(input_img_folder, auxiliary_img_name)
        auxiliary_img = cv2.imread(auxiliary_img_path)
        return auxiliary_img
