import logging
import copy
import cv2
import matplotlib.pyplot as plt

class Builder():
    def __init__(self, params, builder_name):
        self._params = params
        self._img_data = None
        self._annotation_data = None
        self._builder_name = builder_name
        self._print_params()

    def _core_augment_algorithm(self, *args): pass

    def _print_params(self):
        for key, value in self._params.items():
            if key == "allow_execute": continue
            logging.info("--- %s: %s" % (key.upper(), value))

    def _generate_output_name(self, builder_name, params):
        basename, suffix = self._img_data.input_filename.split('.')
        output_img_name = "_".join(
            [basename, builder_name, params]) + "." + suffix
        output_annotation_name = "_".join(
            [basename, builder_name, params]) + ".json"
        self._img_data.output_name = output_img_name
        self._annotation_data.output_name = output_annotation_name
        return output_img_name, output_annotation_name

    def _generate_output_annotation(self, **kwargs):
        new_annotaion = copy.deepcopy(self._annotation_data.input_data)
        new_annotaion['annotation']['filename'] = self._img_data.output_name
        return new_annotaion

    def generate_data(self, *args, **kwargs):
        max_img_output_num = self._params.get('max_img_output_num', 1)
        img_data = []
        ann_data = []
        image_num = [i for i in range(max_img_output_num)]
        for i in image_num:
            output_img_name, output_annotation_name = self._generate_output_name(
                builder_name=self._builder_name, params=str(i))

            self._img_data.output_data = self._core_augment_algorithm()
            self._annotation_data.output_data = self._generate_output_annotation()

            img_data.append((output_img_name, self._img_data.output_data))
            ann_data.append((output_annotation_name, self._annotation_data.output_data))
        return img_data, ann_data

    def feed_data(self, img_data, annotation_data):
        self._img_data = img_data
        self._annotation_data = annotation_data


class ImageData:
    def __init__(self):
        self.filename = None
        self.input_folder = None
        self.input_filename = None
        self.input_data = None
        self.output_folder = None
        self.output_name = None
        self.output_data = None
        self.curr_data_list = []
        self.total_data = []

class AnnotationData:
    def __init__(self):
        self.filename = None
        self.input_folder = None
        self.input_filename = None
        self.input_data = None
        self.output_folder = None
        self.output_name = None
        self.output_data = None
        self.obj_bndbox = None
        self.curr_data_list = []
        self.total_data = []

    def get_objbox(self, data_from='input'):
        if data_from == 'input':
            objects = self.input_data['annotation']['object']
        elif data_from == 'output':
            objects = self.output_data['annotation']['object']
        obj_bndbox = []
        for item in objects:
            bndbox = item['bndbox']
            name = item['name']
            xmin, ymin, xmax, ymax = map(
                int, [bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax']])
            attr = [xmin, ymin, xmax, ymax, name]
            obj_bndbox.append(attr)
        self.obj_bndbox = obj_bndbox
        return obj_bndbox

    def show_border(self, input_data, output_data, annotation_from):
        b, g, r = cv2.split(input_data)
        input_img_rgb = cv2.merge([r, g, b])

        plt.subplot(221)
        plt.imshow(input_img_rgb)

        input_data_ann = copy.deepcopy(input_data)
        obj_bndbox = self.get_objbox()
        for item in obj_bndbox:
            xmin, ymin, xmax, ymax = item[0], item[1], item[2], item[3]
            input_img = cv2.rectangle(input_data_ann, (xmin, ymin), (xmax, ymax), (55, 255, 155), 5)
            b, g, r = cv2.split(input_img)
            input_ann_img_rgb = cv2.merge([r, g, b])
        plt.subplot(222);
        plt.imshow(input_ann_img_rgb)



        b, g, r = cv2.split(output_data)
        output_img_rgb = cv2.merge([r, g, b])

        plt.subplot(223)
        plt.imshow(output_img_rgb)

        output_data_ann = copy.deepcopy(output_data)
        obj_bndbox = self.get_objbox(annotation_from)
        for item in obj_bndbox:
            xmin, ymin, xmax, ymax = item[0], item[1], item[2], item[3]
            output_img = cv2.rectangle(output_data_ann, (xmin, ymin), (xmax, ymax), (55, 255, 155), 5)
            b, g, r = cv2.split(output_img)
            output_ann_img_rbg = cv2.merge([r, g, b])

        plt.subplot(224)
        plt.imshow(output_ann_img_rbg)
        plt.show()

