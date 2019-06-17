# -*- coding:utf-8 -*-
#!/usr/bin/env python

'''
#########################################################
main process for data augment, create more images to classify.
Authors: Dongrun Lu, Yibing Mu, Yiqiao Li, Siwei Liu
Requirements: Python 3.6
#########################################################
'''



import cv2
import json
import logging
import os
import tensorflow as tf
import time
from copy_pasting import CopyPastingBuilder
from image_builder import ImageData, AnnotationData
from mixup import MixupBuilder
from preprocessing import CommonMethodsBuilder
from random_erasing import RandomErasingBuilder


class GenerateController():
    """Enhanced image data controller, create different builders based on params

    Attributes:
        copy_pasting_builder: a builder implemented copy pasting algorithm to enhance image
        mixup_builder: a builder implemented mixup algorithm to enhance image
        random_erasing_builder: a builder implemented random erasing algorithm to enhance image
        common_used_builder: a builder implemented color jittering, affine
                            and etc. to enhance image
        image_data: an ImageData object containing some information related to current image, including
                    image name, image input path, enhanced image output path and etc.
        annotation_data: an AnnotationData object containing some information related to current annotation,
                        including annotation data, bonding box of objects and etc.
    """
    def __init__(self):
        self._copy_pasting_builder = None
        self._mixup_builder = None
        self._random_erasing_builder = None
        self._common_used_builder = None

        self._image_data = None
        self._annotation_data = None

    def _read_img_data(self, img_file_path):
        """Reads image data"""
        return cv2.imread(img_file_path)

    def _read_annotation_data(self, annotation_file_path):
        """Reads annotation data"""
        with open(annotation_file_path, 'r') as json_file:
            annotation_data = json.load(json_file)
            return annotation_data

    @property
    def img_size(self):
        return len(self._image_data.total_data)

    def feed_data(self, img_file_path, annotation_file_path):
        """Feed data to get enhanced image data"""
        self._image_data.total_data = []
        self._annotation_data.total_data = []

        self._image_data.input_data = self._read_img_data(img_file_path)
        self._annotation_data.input_data = self._read_annotation_data(
            annotation_file_path)
        self._image_data.input_filename = os.path.basename(img_file_path)
        self._annotation_data.input_filename = os.path.basename(
            annotation_file_path)

    def set_img_data(self, input_img_folder, output_img_folder):
        """Set image data"""
        self._image_data = ImageData()
        self._image_data.input_folder = input_img_folder
        self._image_data.output_folder = output_img_folder

    def set_annotation_data(self, input_annotation_folder, output_annotation_folder):
        """Set annotation data"""
        self._annotation_data = AnnotationData()
        self._annotation_data.input_folder = input_annotation_folder
        self._annotation_data.output_folder = output_annotation_folder

    def set_builder(self, builder_params):
        """Set the parameters of different builders"""
        self._copy_pasting_parms = builder_params['copy_pasting_params']
        if self._copy_pasting_parms['allow_execute']:
            logging.info("Building Copy Pasting Bulider ...")
            self._copy_pasting_builder = CopyPastingBuilder(self._copy_pasting_parms)

        self._mixup_parms = builder_params['mixup_params']
        if self._mixup_parms['allow_execute']:
            logging.info("Building Mixup Bulider ...")
            self._mixup_builder = MixupBuilder(self._mixup_parms)

        self._random_erasing_parms = builder_params['random_erasing_params']
        if self._random_erasing_parms['allow_execute']:
            logging.info("Building Random Erasing Bulider ...")
            self._random_erasing_builder = RandomErasingBuilder(self._random_erasing_parms)

        self._pre_processing_params = builder_params['pre_processing_params']
        if self._pre_processing_params['allow_execute']:
            logging.info("Building Common used Bulider ...")
            self._common_used_builder = CommonMethodsBuilder(self._pre_processing_params)

    def generate_data(self):
        """Generates image and corresboing annotation from different image builder"""
        builders = [
            self._copy_pasting_builder,
            self._mixup_builder,
            self._random_erasing_builder,
            self._common_used_builder
        ]

        for builder in builders:
            if builder == None: continue
            builder.feed_data(self._image_data, self._annotation_data)
            images, annotations = builder.generate_data()
            self._image_data.total_data += images
            self._annotation_data.total_data += annotations
            self._image_data.curr_data_list = []
            self._annotation_data.curr_data_list = []

        self._save_data()

    def _save_data(self):
        """Saves data and reset image and annotation data"""
        self._save_img()
        self._save_annotation()

    def _save_img(self):
        """Saves image data"""
        self._image_data.total_data.append((self._image_data.input_filename,
                                            self._image_data.input_data))
        for img_file_basename, img_data in self._image_data.total_data:
            output_img_file_path = os.path.join(
                self._image_data.output_folder, img_file_basename)
            cv2.imwrite(output_img_file_path, img_data)

    def _save_annotation(self):
        """Saves annotation data"""
        self._image_data.total_data.append((self._annotation_data.input_filename,
                                            self._annotation_data.input_data))
        for annotation_file_basename, annotation_data in self._annotation_data.total_data:
            output_annotation_file_path = os.path.join(
                self._annotation_data.output_folder, annotation_file_basename)
            annotation_data['annotation']['path'] = os.path.join(
                self._annotation_data.output_folder, annotation_data['annotation']['filename'])
            with open(output_annotation_file_path, 'w') as json_file:
                json.dump(annotation_data, json_file, ensure_ascii=False)


class ImageGenerator():
    """Image Generator

    Attributes:
        input_img_folder: input image data folder
        input_annotation_folder: input annotation data folder
        output_img_folder: output image data folder
        ouput_annotation_folder: output annotation data folder
    """
    def __init__(self, input_folder, output_folder=None):
        self._input_img_folder = input_folder['img_folder']
        self._input_annotation_folder = input_folder['annotation_folder']
        self._output_img_folder = output_folder['img_folder']
        self._output_annotation_folder = output_folder['annotation_folder']

    def set_controller(self, controller):
        """Set enhanced data builder controller"""
        self._controller = controller
        self._controller.set_img_data(self._input_img_folder, self._output_img_folder)
        self._controller.set_annotation_data(self._input_annotation_folder, self._output_annotation_folder)

    def _generate_data(self, input_img_path, input_annotation_path):
        """Feeds data into controller"""
        self._controller.feed_data(input_img_path, input_annotation_path)
        self._controller.generate_data()

    def enhance_data(self):
        """Data augmentation main process"""
        if tf.gfile.Exists(self._output_img_folder):
            tf.gfile.DeleteRecursively(self._output_img_folder)
            tf.gfile.MakeDirs(self._output_img_folder)

        if tf.gfile.Exists(self._output_annotation_folder):
            tf.gfile.DeleteRecursively(self._output_annotation_folder)
            tf.gfile.MakeDirs(self._output_annotation_folder)

        total_num = 0
        walk = tf.gfile.Walk(self._input_img_folder)

        for info in walk:
            img_folder = info[0]
            img_filenames = info[2]
            logging.info("%d images were selected for data augmentation" % len(img_filenames))
            for img_filename in img_filenames:
                if img_filename == '.DS_Store':
                    continue
                img_file_path = os.path.join(img_folder, img_filename)
                annotation_file_path = os.path.join(
                    self._input_annotation_folder, img_filename.split('.')[0] + ".json")

                logging.info("Processing %s ..." % img_file_path)
                start_time = time.time()
                self._generate_data(img_file_path, annotation_file_path)
                end_time = time.time()
                total_num += self._controller.img_size
                logging.info("Finish processing %s, cost %3.3f sec, generate  %d images in total" %
                             (img_file_path, end_time-start_time, total_num))

def main(_):

    input_folder = {
        'img_folder': FLAGS.data_path,
        'annotation_folder': FLAGS.ann_path
    }

    output_folder = {
        'img_folder': FLAGS.enhance_data_path,
        'annotation_folder': FLAGS.enhance_ann_path,
    }

    copy_pasting_params = {
        "allow_execute": FLAGS.copy_pasting,
        "num_object_copied_threshold": FLAGS.copy_pasting_number,
        "max_img_output_num": 10,
        "num_pasting_threshold": 3,
        "mode": FLAGS.copy_pasting_mode
    }

    mixup_params = {
        "allow_execute": FLAGS.mix_up,
        "max_img_output_num": 10,
        "alpha": 0.9
    }

    random_erasing_params = {
        "allow_execute": FLAGS.random_erasing,
        "max_img_output_num": 10
    }

    pre_processing_params = {
        "allow_execute": FLAGS.pre_processing,
        "config_path": FLAGS.config_path
    }

    builder_params = {
        'copy_pasting_params': copy_pasting_params,
        'mixup_params': mixup_params,
        'random_erasing_params': random_erasing_params,
        'pre_processing_params': pre_processing_params
    }

    # Creates an image generator
    image_generator = ImageGenerator(input_folder, output_folder)

    # Builds a controller to control builders
    generate_controller = GenerateController()
    generate_controller.set_builder(builder_params)

    # Choose one controller as a main generator
    image_generator.set_controller(generate_controller)
    image_generator.enhance_data()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s: %(message)s',
                        level=logging.DEBUG)

    # default_data_path = "/Users/ryanlu/PycharmProjects/LearnTensorflow/ImageProcessing/test_img/"
    # default_enhance_data_path = "/Users/ryanlu/PycharmProjects/LearnTensorflow/ImageProcessing/enhance/"
    # default_ann_path = "/Users/ryanlu/PycharmProjects/LearnTensorflow/ImageProcessing/annotations/"
    # default_enhance_ann_path = '/Users/ryanlu/PycharmProjects/LearnTensorflow/ImageProcessing/enhance_ann/'
    default_data_path = "/Users/ryanlu/Downloads/YuQianmu_traindata_3_30/JPEGImages/"
    default_enhance_data_path = "/Users/ryanlu/Downloads/YuQianmu_traindata_3_30/enhance/"
    default_ann_path = "/Users/ryanlu/Downloads/YuQianmu_traindata_3_30/Annotations/"
    default_enhance_ann_path = '/Users/ryanlu/Downloads/YuQianmu_traindata_3_30/enhance_ann/'
    default_config_path = './config.json'

    default_copy_pasting = False
    default_copy_pasting_mode = 'Multiple'
    default_copy_pasting_number = 3

    default_preprocessing = True
    default_mixup = True
    default_random_erasing = True

    # define the main path
    tf.app.flags.DEFINE_string(
        'data_path', default_data_path, 'original image data path')
    tf.app.flags.DEFINE_string(
        'enhance_data_path', default_enhance_data_path, 'path for storing enhanced image data')
    tf.app.flags.DEFINE_string(
        'ann_path', default_ann_path, 'annotation of original image data path')
    tf.app.flags.DEFINE_string(
        'enhance_ann_path', default_enhance_ann_path, 'path for storing enhanced annotation')
    tf.app.flags.DEFINE_string('config_path', default_config_path,
                               'pamameters of image preprocessing methods config json path')

    # define the mode of copy pasting
    tf.app.flags.DEFINE_boolean(
        'copy_pasting', default_copy_pasting, 'copy pasting data augment')
    tf.app.flags.DEFINE_string(
        'copy_pasting_mode', default_copy_pasting_mode, 'copy pasting mode')
    tf.app.flags.DEFINE_integer(
        'copy_pasting_number', default_copy_pasting_number, 'copy pasting number (up to 10)')

    # define the parameters of image preprocessing
    tf.app.flags.DEFINE_boolean('pre_processing', default_preprocessing,
                                'preprocessing methods like adjusting brightness, hue and etc.')
    tf.app.flags.DEFINE_boolean('mix_up', default_mixup, 'mixup data augment')
    tf.app.flags.DEFINE_boolean(
        'random_erasing', default_random_erasing, 'mixup data augment')

    FLAGS = tf.app.flags.FLAGS
    tf.app.run()
