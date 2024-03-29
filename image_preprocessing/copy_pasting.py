import cv2
import random
import math
import copy
import numpy as np
from image_builder import Builder

class CopyPastingBuilder(Builder):
    """ Copy pasting algorithm for data augment

    Implemente the code for copy pasting algorithm. For default, create three copies for each small object.
    For each copy, It has a random rotate angle and we choose a random place to paste it. Further more, these
    objects generated by functions will not be overlapped with each other.
    """
    def __init__(self, params):
        Builder.__init__(self, params, builder_name='copy_pasting')
        self._num_object_copied_threshold = params['num_object_copied_threshold']
        self._num_pasting_threold = params['num_pasting_threshold']
        self._mode = params['mode']
        self._update_obj_bndbox = None

    def _core_augment_algorithm(self):
        """Implements copy pasting algorithm"""
        new_obj_bndbox = copy.deepcopy(self._annotation_data.get_objbox())
        img_data = copy.deepcopy(self._img_data.input_data)

        bound = 0.0123
        originArea = img_data.shape[0] * img_data.shape[1] * bound
        candidates = []

        for item in new_obj_bndbox:
            area = abs(item[0] - item[2]) * abs(item[1] - item[3])
            if (area < originArea):
                candidates.append(item)
                candidates.append(item)

        np.random.shuffle(candidates)

        if self._mode == 'Single':
            candidates = [candidates[0]]
        else:
            if len(candidates) > 3:
                candidates = candidates[:2]
        
        # step2: rotate and copy pasting
        for cur in candidates:
            copy_pasting_num = 0
            while copy_pasting_num < self._num_pasting_threold:
                assert copy_pasting_num < 10
                copy_pasting_num += 1
                imgNew = self._rotate_part(img_data, cur)
                imageNewHeight = imgNew.shape[0]
                imageNewWidth = imgNew.shape[1]

                tryTime = 0
                while tryTime < 10:
                    tryTime += 1
                    rx = random.randint(0, img_data.shape[1])
                    ry = random.randint(0, img_data.shape[0])
                    newBound = [rx, ry, rx + imageNewWidth,
                                ry + imageNewHeight, cur[4]]

                    flag = False
                    # check if it is not out of origin image bound
                    if newBound[3] <= img_data.shape[0] and newBound[2] <= img_data.shape[1]:
                        for item1 in new_obj_bndbox:
                            if self._is_overlap(newBound, item1):
                                flag = True
                                break

                        if not flag:
                            new_obj_bndbox.append(newBound)

                            ticket = imgNew[0:imageNewHeight, 0:imageNewWidth]
                            # TODO(Yibing Mu) Fix the thin border bug.

                            height = slice(newBound[1], newBound[3])
                            width = slice(newBound[0], newBound[2])

                            i = 0
                            for h_pix in range(height.start, height.stop):
                                j = 0
                                for w_pix in range(width.start, width.stop):
                                    if ticket[i][j][-1] != 0:
                                        for k in range(3):
                                            img_data[h_pix][w_pix][k] = ticket[i][j][k]
                                    j += 1
                                i += 1

                            # img = cv2.rectangle(img, (newBound[0], newBound[1]), (newBound[2], newBound[3]),(55, 255, 155), 3)
                            break
        self._update_obj_bndbox = new_obj_bndbox

        return img_data

    def _generate_output_annotation(self):
        """Generates output annotation"""
        new_annotaion = copy.deepcopy(self._annotation_data.input_data)
        new_annotaion['annotation']['object'] = []
        for attr in self._update_obj_bndbox:
            obj = {
                'bndbox': {
                    'xmin': attr[0],
                    'ymin': attr[1],
                    'xmax': attr[2],
                    'ymax': attr[3]
                },
                'name': attr[4]
            }
            new_annotaion['annotation']['object'].append(obj)
        new_annotaion['annotation']['filename'] = self._img_data.output_name
        return new_annotaion

    def _is_overlap(self, new, origin):
        """ Check if two objects are overlapped.
        Args:
            new: rotated image data
            origin: original image data
        Return:
            boolean: true if two objects are overlapped
        """
        left = max(new[0], origin[0])
        right = min(new[2], origin[2])
        ymin = max(new[1], origin[1])
        ymax = min(new[3], origin[3])
        if left < right and ymin < ymax:
            return True
        else:
            return False

    def _rotate_part(self, img, point=None):
        """Rotate target object with random angle
        Args:
            img: image data
            point: 4 vertices of bonding box: [xmin, ymin, xmax, ymax]
        Return:
            n rotated image
        """
        degree = random.randint(0, 360)
        xmin = point[0]
        ymin = point[1]
        xmax = point[2]
        ymax = point[3]

        copyImg = img[ymin:ymax, xmin:xmax]

        # add alpha chanel for the object image. We need to remove the affect of
        # black padding.  This wil tell us which is object, which is padding.
        # It will be used to replace original image
        b_channel, g_channel, r_channel = cv2.split(copyImg)
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
        img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        height, width = img_BGRA.shape[:2]

        heightNew = int(width * math.fabs(math.sin(math.radians(degree))) + height * math.fabs(
            math.cos(math.radians(degree))))
        widthNew = int(height * math.fabs(math.sin(math.radians(degree))) + width * math.fabs(
            math.cos(math.radians(degree))))
        M = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
        M[0, 2] += (widthNew - width) / 2
        M[1, 2] += (heightNew - height) / 2
        imgRotation = cv2.warpAffine(
            img_BGRA, M, (widthNew, heightNew), borderValue=(255, 255, 255, 0))
        return imgRotation