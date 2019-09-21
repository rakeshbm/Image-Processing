# -*- coding:utf-8 -*-
# !/usr/bin/env python

'''
#########################################################
enhance image data by multiple image pre-processing methods, create more pcitures to classify.
Authors: Dongrun Lu, Yibing Mu, Yiqiao Li, Siwei Liu
Requirements: Python 3.6
#########################################################
'''

import cv2
import imutils
import json
import math
import os
import re
import random
import numpy as np
import tensorflow as tf
from scipy import ndimage


class ImagePreprocessing:
    """ Image preprocessing methods for data augment

    There are too many image preprocessing methods such as geometric transformation and color jittering.
    If you want to control the output, please modify the 'config.json'

    Attributes:
        ann_path: annotations path
        ann: whether or not containing annotations
        enhance_data_path: data path for saving enhanced image
        process_config_path: parameters of image preprocessing methods
    """

    def __init__(self, enhance_data_path, ann_path='', enhance_ann_path='', process_config_path=''):
        self.ann_path = ann_path
        self.ann = False if self.ann_path == '' else True
        self.enhance_ann_path = enhance_ann_path
        self.process_config = self.load_config(process_config_path)
        self.print_parameters()

    def load_config(self, process_config_path):
        """
        load parameters of image pre-processing from 'config.json'
        """
        with open(process_config_path, 'r') as json_file:
            config = json.load(json_file)
            return config

    def print_parameters(self):
        """
        print the methods in use
        """
        for key in self.process_config:
            print("-- %s" % key)
        print("")

    def EnhancePictureAndSave(self, picname, savepath):
        '''
        use [tf.image.flip_up_down, tf.image.flip_left_right, tf.image.adjust_brightness,
        tf.image.adjust_contrast, tf.image.adjust_hue, tf.image.adjust_saturation, tf.image.adjust_gamma] and some
        openCV image preprocessing methods to enhance origin picture, and save to enhance data path
        '''

        tf.reset_default_graph()

        filename, suffix = os.path.splitext(picname)  # get picture path
        filename = os.path.basename(filename)  # get base picture name
        filename += "_"

        image = tf.read_file(picname)  # read picture from gving path.
        image_decode_jpeg = tf.image.decode_jpeg(image)
        image_decode_jpeg = tf.image.convert_image_dtype(image_decode_jpeg,
                                                         dtype=tf.float32)
        # adjust brightness
        if self.process_config.get('bright'):
            delta = self.process_config['bright']
            image_random_brightness = tf.image.adjust_brightness(image_decode_jpeg, delta=delta)
            para_desc = "_with_delta_%s" % (delta)
            self.tf_augment(image_random_brightness, savepath, filename, suffix, 'bright', para_desc)

        # adjust contrast
        if self.process_config.get('contrast'):
            delta = self.process_config['contrast']
            image_random_contrast = tf.image.adjust_contrast(image_decode_jpeg, delta)
            para_desc = "_with_delta_%s" % (delta)
            self.tf_augment(image_random_contrast, savepath, filename, suffix, 'contrast', para_desc)

        # adjust hue
        if self.process_config.get('hue'):
            delta = self.process_config['hue']
            image_random_hue = tf.image.adjust_hue(image_decode_jpeg, delta=delta)
            para_desc = "_with_delta_%s" % (delta)
            self.tf_augment(image_random_hue, savepath, filename, suffix, 'hue', para_desc)

        # adjust saturation
        if self.process_config.get('saturation'):
            delta = self.process_config['saturation']
            image_random_saturation = tf.image.adjust_saturation(image_decode_jpeg, delta)
            para_desc = "_with_delta_%s" % (delta)
            self.tf_augment(image_random_saturation, savepath, filename, suffix, 'saturation', para_desc)

        # adjust gamma
        if self.process_config.get('gamma'):
            delta = self.process_config['gamma']
            image_adjust_gamma = tf.image.adjust_gamma(image_decode_jpeg, gamma=delta)
            para_desc = "_with_delta_%s" % (delta)
            self.tf_augment(image_adjust_gamma, savepath, filename, suffix, 'gamma', para_desc)

        tf.get_default_graph().finalize()

        # flip horizontal
        cvImg = cv2.imread(picname)
        cvImgHorizontalList = self.horizontal_flipping(cvImg)
        self.saveCvImage(cvImgHorizontalList, "filp_horizontal", filename, suffix, savepath)

        # flip vertical
        cvImg = cv2.imread(picname)
        cvImgVerticalList = self.vertical_flipping(cvImg)
        self.saveCvImage(cvImgVerticalList, "filp_vertical", filename, suffix, savepath)

        # rotation
        cvImg = cv2.imread(picname)
        cvImgRotationList = self.rotation(cvImg)
        self.saveCvImage(cvImgRotationList, "rotation", filename, suffix, savepath)

        # histogram equalization
        cvImg = cv2.imread(picname)
        cvImgHisEqulColorList = self.his_equl_color(cvImg)
        self.saveCvImage(cvImgHisEqulColorList, "hisEqulColor", filename, suffix, savepath)

        # poisson noise
        cvImg = cv2.imread(picname)
        cvImgPoissonNoiseList = self.poisson_noise(cvImg)
        self.saveCvImage(cvImgPoissonNoiseList, "poissonNoise", filename, suffix, savepath)
        
        # speckle noise
        cvImg = cv2.imread(picname)
        cvImgSpeckleNoiseList = self.speckle_noise(cvImg)
        self.saveCvImage(cvImgSpeckleNoiseList, "speckleNoise", filename, suffix, savepath)
        
        # affine transform
        cvImg = cv2.imread(picname)
        cvImgAffineTransformList = self.affine_transform(cvImg)
        self.saveCvImage(cvImgAffineTransformList, "affineTransform", filename, suffix, savepath)
        
        # adjust scale
        if self.process_config['scale']:
            cvImg = cv2.imread(picname)
            cvImgScaleList = self.scale(cvImg)
            self.saveCvImage(cvImgScaleList, "scale", filename, suffix, savepath)
        
        # translation
        if self.process_config['translation']:
            cvImg = cv2.imread(picname)
            cvImgTranslationList = self.translation(cvImg)
            self.saveCvImage(cvImgTranslationList, "translation", filename, suffix, savepath)
            
        # adjust sharpening
        if self.process_config['sharpening']:
            cvImg = cv2.imread(picname)
            cvImgSharpeningList = self.sharpening(cvImg)
            self.saveCvImage(cvImgSharpeningList, "sharpening", filename, suffix, savepath)

        # adjust median blur
        if self.process_config['median_blur']:
            cvImg = cv2.imread(picname)
            cvImgMedianBlurList = self.median_blur(cvImg)
            self.saveCvImage(cvImgMedianBlurList, "medianBlur", filename, suffix, savepath)

        # adjust bilateral blur
        if self.process_config['bilateral_blur']:
            cvImg = cv2.imread(picname)
            cvImgBilateralBlurList = self.bilateral_blur(cvImg)
            self.saveCvImage(cvImgBilateralBlurList, "bilateralBlur", filename, suffix, savepath)

        # denoising
        if self.process_config['denoising']:
            cvImg = cv2.imread(picname)
            cvImgDenoisingList = self.denoising(cvImg)
            self.saveCvImage(cvImgDenoisingList, "denoising", filename, suffix, savepath)

        # guassian blurr
        if self.process_config['guassian']:
            cvImg = cv2.imread(picname)
            cvImgGuassianBlurList = self.guassian_blur(cvImg)
            self.saveCvImage(cvImgGuassianBlurList, "guassianBlur", filename, suffix, savepath)
            
        # salt and pepper noise
        if self.process_config['salt_and_pepper']:
            cvImg = cv2.imread(picname)
            cvImgSaltAndPepperList = self.salt_and_pepper_noise(cvImg)
            self.saveCvImage(cvImgSaltAndPepperList, "saltAndPepperNoise", filename, suffix, savepath)

    def tf_augment(self, image, savepath, filename, suffix, function, paramater):
        """
        save the enhanced image generated by tensorflow methods
        """
        convert_image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        convert_image = tf.image.encode_jpeg(convert_image)
        openfile = filename + function + paramater + suffix
        hd_adj = tf.gfile.FastGFile(os.path.join(savepath, openfile), "w")

        with tf.Session() as sess:  # create tensorflow session
            convert_image_1 = sess.run(convert_image)
            sess.close()

        hd_adj.write(convert_image_1)
        hd_adj.close()

        self.saveAnn(openfile, filename, function, savepath)

    def saveCvImage(self, imageList, functionname, filename, suffix, savepath):
        """
        save the enhanced image generated by opencv methods
        """
        for i in range(len(imageList)):
            parameter = imageList[i][1]
            file = imageList[i][0]
            openfile = filename + "_" + functionname + "_" + parameter + suffix
            cv2.imwrite(os.path.join(savepath, openfile), file)
            if self.ann == True:
                self.saveAnn(openfile, filename, functionname, savepath)

    def saveAnn(self, openfile, filename, functionname, savepath):
        """
        save the annotations after data augment
        """
        img_ann_path = self.ann_path + filename[:-1] + '.json'
        with open(img_ann_path, 'r') as json_file:
            ann = json.load(json_file)
            obj = ann['annotation']['object']

            img_size = ann['annotation']['size']
            img_width = int(img_size['width'])
            img_height = int(img_size['height'])

            for i, item in enumerate(obj):
                bndbox = item['bndbox']
                xmin = int(bndbox['xmin'])
                ymin = int(bndbox['ymin'])
                xmax = int(bndbox['xmax'])
                ymax = int(bndbox['xmax'])
                point = [xmin, ymin, xmax, ymax]

                update_parameter = False
                if functionname == 'rotation':
                    degree = int(openfile.split('degree')[0].split('rotate')[1])
                    new_img_width, new_img_height, new_xmin, new_ymin, new_xmax, new_ymax = self.rotation_point(
                        img_width, img_height, angle=degree, point=point)
                    update_parameter = True
                elif functionname == 'scale':
                    para = re.findall('([0-9]\.[0-9])', openfile)
                    x_scale = float(para[0])
                    y_scale = float(para[1])
                    new_img_width, new_img_height, new_xmin, new_ymin, new_xmax, new_ymax = self.ann_rescale(img_width,
                                                                                                             img_height,
                                                                                                             point,
                                                                                                             x_scale,
                                                                                                             y_scale)
                    update_parameter = True
                elif functionname == 'filp_horizontal' or functionname == 'filp_vertical':
                    new_img_width, new_img_height, new_xmin, new_ymin, new_xmax, new_ymax = self.ann_flipping(img_width,
                                                                                                              img_height,
                                                                                                              functionname,
                                                                                                              point)
                    update_parameter = True

                if update_parameter == True:
                    ann['annotation']['object'][i]['bndbox']['xmin'] = new_xmin
                    ann['annotation']['object'][i]['bndbox']['ymin'] = new_ymin
                    ann['annotation']['object'][i]['bndbox']['xmax'] = new_xmax
                    ann['annotation']['object'][i]['bndbox']['ymax'] = new_ymax
                    ann['annotation']['size']['width'] = new_img_width
                    ann['annotation']['size']['height'] = new_img_height

                ann['annotation']['path'] = os.path.join(savepath, openfile)
                ann['annotation']['filename'] = openfile

            openjson = self.enhance_ann_path + openfile.split('.jpg')[0] + '.json'
            with open(openjson, 'w') as json_file:
                json.dump(ann, json_file, ensure_ascii=False)

    def ann_rescale(self, img_width, img_height, point, w, h):
        xmin, xmax, ymin, ymax = point[0], point[2], point[1], point[3]
        return int(img_width * w), int(img_height * h), int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)

    def rotation_point(self, img_width, img_height, angle=45, point=None, cut_edge=False):
        """
        Generate size of a rotated image and the vertex coordinates of correspoing bonding boxes

        Args:
            img_width: width of an image
            img_height: height of an image
            angle: the rotation angle of the image
            point: the four vertex coordinates of bonding box
            cut_egde: whether to preserve the size of image after rotating

        Returns:
            size of a rotated image,
            the vertex coordinates of correspoing bonding boxes

        """
        pi = 3.14159
        # angle = 360 - angle % 360
        nx = img_width
        ny = img_height
        xmin, xmax, ymin, ymax = point[0], point[2], point[1], point[3]
        point1 = [xmin, ymax]
        point2 = [xmax, ymax]
        point3 = [xmax, ymin]
        point4 = [xmin, ymin]
        x2 = int(img_width / 2)
        y2 = int(img_height / 2)
        point = [point1, point2, point3, point4]
        cornerPoint = [[0, 0], [img_width, 0], [img_width, img_height], [0, img_height]]
        newpoint = []
        newCornerPoint = []
        for item in point:
            x1 = item[0];
            y1 = item[1];
            x = (x1 - x2) * math.cos(pi / 180.0 * angle) - (y1 - y2) * math.sin(pi / 180.0 * angle) + x2;
            y = (x1 - x2) * math.sin(pi / 180.0 * angle) + (y1 - y2) * math.cos(pi / 180.0 * angle) + y2;
            newpoint.append([x, y])

        for item in cornerPoint:
            x1 = item[0];
            y1 = item[1];
            x = (x1 - x2) * math.cos(pi / 180.0 * angle) - (y1 - y2) * math.sin(pi / 180.0 * angle) + x2;
            y = (x1 - x2) * math.sin(pi / 180.0 * angle) + (y1 - y2) * math.cos(pi / 180.0 * angle) + y2;
            newCornerPoint.append([x, y])
        moreX = 0
        moreY = 0
        if cut_edge == False:
            moreX = abs(
                int(min(newCornerPoint[0][0], newCornerPoint[1][0], newCornerPoint[2][0], newCornerPoint[3][0])))
            moreY = abs(
                int(min(newCornerPoint[0][1], newCornerPoint[1][1], newCornerPoint[2][1], newCornerPoint[3][1])))
            nx += moreX
            ny += moreY

        newXmin = max(int(min(newpoint[0][0], newpoint[1][0], newpoint[2][0], newpoint[3][0])) + moreX, 0)
        newXmax = min(int(max(newpoint[0][0], newpoint[1][0], newpoint[2][0], newpoint[3][0])) + moreX, nx)
        newYmin = max(int(min(newpoint[0][1], newpoint[1][1], newpoint[2][1], newpoint[3][1])) + moreY, 0)
        newYmax = min(int(max(newpoint[0][1], newpoint[1][1], newpoint[2][1], newpoint[3][1])) + moreY, ny)
        return img_width, img_height, newXmin, newYmin, newXmax, newYmax

    def ann_flipping(self, img_width, img_height, functionname, point=None):
        if functionname == 'filp_horizontal':
            return img_width, img_height, img_width - point[2], point[1], img_width - point[0], point[3]
        else:
            return img_width, img_height, point[0], img_height - point[3], point[2], img_height - point[1]

    def his_equl_color(self, img):
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        cv2.equalizeHist(channels[0], channels[0])
        cv2.merge(channels, ycrcb)
        cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
        return [(img, "Histogram")]

    def horizontal_flipping(self, image):
        flip_horizontal = cv2.flip(image, 0)
        return [(flip_horizontal, "horizontal_flip")]

    def vertical_flipping(self, image):
        flip_horizontal = cv2.flip(image, 1)
        return [(flip_horizontal, "vertical_flip")]

    def rotation(self, image):
        rotation_list = []
        cut_edge = False
        for angle in np.arange(45, 360, 15):
            if cut_edge == False:
                angle = 360 - angle
                rotated = imutils.rotate_bound(image, angle)
            else:
                rotated = imutils.rotate(image, angle)
            rotation_list.append((rotated, "rotate" + str(angle) + "degree"))
        return rotation_list
    
    def translation(self, image):
        rst = []
        num_rows, num_cols = image.shape[:2]
        for item in self.process_config['translation']:
            translation_matrix = np.float32(item.translation_matrix)
            img_translation = cv2.warpAffine(image, translation_matrix, (num_cols, num_rows))
            rst.append((img_translation, "translation"))
        return rst

    def sharpening(self, image):
        rst = []
        for item in self.process_config['sharpening']:
            first_filter = int(item['first_filter'])
            second_filter = int(item['second_filter'])
            alpha = int(item['alpha'])

            blurred_f = ndimage.gaussian_filter(image, first_filter)
            filter_blurred_f = ndimage.gaussian_filter(blurred_f, second_filter)
            rst_img = blurred_f + alpha * (blurred_f - filter_blurred_f)

            para_desc = "Sharpened_with_(%d_%d_%d)" % (first_filter, second_filter, alpha)
            rst.append((rst_img, para_desc))
        return rst

    def guassian_blur(self, image):
        rst = []
        for item in self.process_config['guassian']:
            width = int(item['gamma_width'])
            height = int(item['gamma_height'])
            guassian_blur = int(item['gamma_blur'])
            rst_img = cv2.GaussianBlur(image, (width, height), guassian_blur)
            para_desc = "gussian_with_(%s_%s_%s)" % (width, height, guassian_blur)
            rst.append((rst_img, para_desc))
        return rst

    def denoising(self, image):
        rst = []
        for item in self.process_config['denoising']:
            filterLength = int(item['filter_length'])
            colorComponent = int(item['color_component'])
            tempWindow = int(item['temp_window_size'])
            searchWindow = int(item['search_window_size'])
            rst_img = cv2.fastNlMeansDenoisingColored(image, None, filterLength, colorComponent, tempWindow,
                                                      searchWindow)
            para_desc = "denoising_with_filter_strength_(%d_%d_%d_%d)" % (
                filterLength, colorComponent, tempWindow, searchWindow)
            rst.append((rst_img, para_desc))
        return rst

    def median_blur(self, image):
        rst = []
        for item in self.process_config['median_blur']:
            median_blur_strength = int(item['median_blur_strength'])
            rst_img = cv2.medianBlur(image, median_blur_strength)
            para_desc = "median_blur_(%d)" % (median_blur_strength)
            rst.append((rst_img, para_desc))
        return rst

    def bilateral_blur(self, image):
        rst = []
        for item in self.process_config['bilateral_blur']:
            diameter = int(item['diameter'])
            sigma_color = int(item['sigma_color'])
            sigma_space = int(item['sigma_space'])

            rst_img = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
            para_desc = "bilateral_blur_(%d_%d_%d)" % (diameter, sigma_color, sigma_space)
            rst.append((rst_img, para_desc))
        return rst

    def scale(self, image):
        rst = []
        for item in self.process_config['scale']:
            x = float(item['x'])
            y = float(item['y'])
            rst_img = cv2.resize(image, None, fx=x, fy=y, interpolation=cv2.INTER_CUBIC)
            para_desc = "rescale-X_%s_rescale-Y_%s" % (x, y)
            rst.append((rst_img, para_desc))
        return rst
    
    def salt_and_pepper_noise(self, image):
        rst  = []
        row,col,ch = image.shape
        
        for item in self.process_config['salt_and_pepper']:
            rst_img = np.copy(image)
            s_vs_p = float(item['s_vs_p'])
            amount = float(item['amount'])
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            rst_img[coords] = 1
        
            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            rst_img[coords] = 0
            para_desc = "salt-vs-pepper_%s_amount_%s" % (s_vs_p, amount)
            rst.append((rst_img, para_desc))
        return rst
    
    def poisson_noise(self, image):
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return [(noisy, "poisson_noise")]
    
    def speckle_noise(self, image):
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return [(noisy, "speckle_noise")]

    def affine_transform(self, image):
        rows, cols, ch = image.shape
        cv2.circle(image, (83, 90), 5, (0, 0, 255), -1)
        cv2.circle(image, (447, 90), 5, (0, 0, 255), -1)
        cv2.circle(image, (83, 472), 5, (0, 0, 255), -1)
        pts1 = np.float32([[83, 90], [447, 90], [83, 472]])
        pts2 = np.float32([[0, 0], [447, 90], [150, 472]])
        matrix = cv2.getAffineTransform(pts1, pts2)
        rst_img = cv2.warpAffine(image, matrix, (cols, rows))
        return [(rst_img, "affine_transform")]
        
def test():
    print("begin to enhance picture data!!!")

    image_path = "/Users/ryanlu/PycharmProjects/LearnTensorflow/ImageProcessing/test_img/16.jpg"
    ann_path = "/Users/ryanlu/PycharmProjects/LearnTensorflow/ImageProcessing/annotations/"
    enhance_data_path = "/Users/ryanlu/PycharmProjects/LearnTensorflow/ImageProcessing/enhance/"
    enhance_ann_path = '/Users/ryanlu/PycharmProjects/LearnTensorflow/ImageProcessing/enhance_ann/'
    config_path = './config.json'

    ann_flag = True
    if ann_flag == False:
        ip = ImagePreprocessing(enhance_data_path, process_config_path=config_path)
        ip.EnhancePictureAndSave(image_path, enhance_data_path)
    else:
        ip = ImagePreprocessing(enhance_data_path, ann_path, enhance_ann_path, config_path)
        ip.EnhancePictureAndSave(image_path, enhance_data_path)

    print("end of enhance picture data, good luck!!!")


if __name__ == "__main__":
    test()
