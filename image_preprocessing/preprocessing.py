from image_builder import Builder
import cv2
import copy
import json
import logging
import numpy as np
import imutils
import math
from scipy import ndimage


class CommonMethodsBuilder(Builder):
    def __init__(self, params):
        self._methods_params = self._load_config(params['config_path'])
        Builder.__init__(self, self._methods_params, builder_name='mix_up')
        self._update_ann_func_name = ('rotation', 'horizontal_flipping',
                                      'scale', 'vertical_flipping',
                                      'perspective_transform')

    def generate_data(self, *args, **kwargs):
        self._img_data.curr_data_list = []
        self._annotation_data.curr_data_list = []

        self._generate_output(enhance_func=self._horizontal_flipping)
        self._generate_output(enhance_func=self._vertical_flipping)
        self._generate_output(enhance_func=self._histogram_equalization)
        self._generate_output(enhance_func=self._rotation,
                              params=[angle for angle in np.arange(45, 360, 30)])

        if self._params.get('denoising'):
            self._generate_output(
                enhance_func=self._denoising, params=self._params.get('denoising'))
        if self._params.get('guassian'):
            self._generate_output(
                enhance_func=self._guassian_blur, params=self._params.get('guassian'))
        if self._params.get('median_blur'):
            self._generate_output(
                enhance_func=self._median_blur, params=self._params.get('median_blur'))
        if self._params.get('bilateral_blur'):
            self._generate_output(
                enhance_func=self._bilateral_blur, params=self._params.get('bilateral_blur'))
        if self._params.get('scale'):
            self._generate_output(
                enhance_func=self._scale, params=self._params.get('scale'))
        if self._params.get('sharpening'):
            self._generate_output(
                enhance_func=self._sharpening, params=self._params.get('sharpening'))
        if self._params.get('bright'):
            self._generate_output(
                enhance_func=self._bright, params=self._params.get('bright'))
        if self._params.get('contrast'):
            self._generate_output(
                enhance_func=self._contrast, params=self._params.get('contrast'))
        if self._params.get('perspective_transform'):
            self._generate_output(
                enhance_func=self._perspective_transform, params=self._params.get('perspective_transform'))


        return self._img_data.curr_data_list, self._annotation_data.curr_data_list

    def _load_config(self, process_config_path):
        """load parameters of image pre-processing from 'config.json'"""
        with open(process_config_path, 'r') as json_file:
            config = json.load(json_file)
            return config

    def _generate_output(self, enhance_func, params=None):
        if params == None:
            params = ['']
        func_name = enhance_func.__name__[1:]

        for param in params:
            img_data = copy.deepcopy(self._img_data.input_data)
            # generate enhanced image data
            self._img_data.output_data, curr_param = enhance_func(img_data, param)
            # generate output names
            output_img_name, output_annotation_name = self._generate_output_name(func_name, curr_param)
            # generate annotation output
            output_annotation_data = self._generate_output_annotation(methods_name=func_name, param=param)

            self._img_data.curr_data_list.append((output_img_name, self._img_data.output_data))
            self._annotation_data.curr_data_list.append((output_annotation_name, output_annotation_data))

    def _generate_output_annotation(self, **kwargs):
        new_annotaion = copy.deepcopy(self._annotation_data.input_data)
        new_annotaion['annotation']['filename'] = self._img_data.output_name
        func_name = kwargs['methods_name']
        if func_name not in self._update_ann_func_name:
            return new_annotaion

        param = kwargs['param']
        obj = new_annotaion['annotation']['object']
        img_size = new_annotaion['annotation']['size']
        img_width = int(img_size['width'])
        img_height = int(img_size['height'])
        if type(obj) == dict:
            obj = [obj]
            new_annotaion['annotation']['object'] = [new_annotaion['annotation']['object']]
        for i, item in enumerate(obj):
            bndbox = item['bndbox']
            xmin, ymin = int(bndbox['xmin']), int(bndbox['ymin'])
            xmax, ymax = int(bndbox['xmax']), int(bndbox['ymax'])
            point = [xmin, ymin, xmax, ymax]

            update_img_info, update_points = None, None
            if func_name == 'rotation':
                angle = 360 - param
                update_img_info, update_points = self._get_rotation_ann(
                    img_width, img_height,angle=angle, point=point)

            elif func_name == 'scale':
                x_scale, y_scale = param['x'], param['y']
                update_img_info, update_points = self._get_rescale_ann(
                    img_width, img_height,point, x_scale, y_scale)

            elif func_name == 'horizontal_flipping' or func_name == 'vertical_flipping':
                update_img_info, update_points = self._get_flipping_ann(
                    img_width, img_height, func_name, point)
            elif func_name == 'perspective_transform':
                update_img_info, update_points = self._get_pers_transform_ann(
                    img_width, img_height, param['p'], param['flag'], point)

            new_img_width, new_img_height = update_img_info
            new_xmin, new_ymin, new_xmax, new_ymax = update_points
            new_annotaion['annotation']['object'][i]['bndbox']['xmin'] = new_xmin
            new_annotaion['annotation']['object'][i]['bndbox']['ymin'] = new_ymin
            new_annotaion['annotation']['object'][i]['bndbox']['xmax'] = new_xmax
            new_annotaion['annotation']['object'][i]['bndbox']['ymax'] = new_ymax
            new_annotaion['annotation']['size']['width'] = new_img_width
            new_annotaion['annotation']['size']['height'] = new_img_height

        # if func_name == 'perspective_transform':
        #     self._annotation_data.output_data = new_annotaion
        #     self._annotation_data.show_border(self._img_data.input_data,
        #                                       self._img_data.output_data,
        #                                       annotation_from='output')
        return new_annotaion

    def _get_rescale_ann(self, img_width, img_height, point, w, h):
        xmin, xmax, ymin, ymax = point[0], point[2], point[1], point[3]
        return ((int(img_width * w), int(img_height * h)),
                (int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)))

    def _get_pers_transform_ann(self, img_width, img_height, p, flag, point):
        rows, cols = img_height, img_width
        # Xmin ymin xmax ymax
        xmin, xmax, ymin, ymax = point[0], point[2], point[1], point[3]

        x = np.array([[xmin, ymin, 1], [xmin, ymax, 1], [xmax, ymax, 1], [xmax, ymin, 1]])
        x = x.T
        pts1 = np.float32([[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]])

        if flag == 1:
            view = np.float32([[-cols * p, 0], [cols * (1 + p), 0], [474, rows], [0, rows]])
        elif flag == 2:
            view = np.float32([[0, 0], [cols, 0], [cols * (1 + p), rows], [-cols * p, rows]])
        elif flag == 3:
            view = np.float32([[0, -rows * p], [cols, 0], [cols, rows], [0, rows * (1 + p)]])
        else:
            view = np.float32([[0, 0], [cols, -rows * p], [cols, rows * (1 + p)], [0, rows]])

        M = cv2.getPerspectiveTransform(pts1, view)
        res = np.dot(M, x)
        x_arr = res[0] / res[2]
        y_arr = res[1] / res[2]

        newXmin = max(int(min(x_arr[0], x_arr[1])), 0)
        newXmax = min(int(max(x_arr[2], x_arr[3])), int(rows - 1))
        newYmin = max(int(min(y_arr[0], y_arr[3])), 0)
        newYmax = min(int(max(y_arr[1], y_arr[2])), int(cols - 1))
        return ((img_width, img_height),
                (newXmin, newYmin, newXmax, newYmax))

    def _get_flipping_ann(self, img_width, img_height, functionname, point=None):
        img_info = (img_width, img_height)
        if functionname == 'vertical_flipping':
            points = (img_width - point[2], point[1], img_width - point[0], point[3])
        else:
            points = (point[0], img_height - point[3], point[2], img_height - point[1])
        return (img_info, points)

    def _bright(self, image, params):
        value = params['delta']
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        rst_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        para_desc = "(+%d)" % (value) if value >= 0 else "(-%d)" % (value)
        return rst_img, para_desc

    def _contrast(self, image, params):
        alpha = params['delta']
        rst_img = np.uint8(np.clip((alpha * image), 0, 255))
        para_desc = "(%3.3f)" % (alpha)
        return rst_img, para_desc


    def _sharpening(self, image, params):
        first_filter = int(params['first_filter'])
        second_filter = int(params['second_filter'])
        alpha = int(params['alpha'])

        blurred_f = ndimage.gaussian_filter(image, first_filter)
        filter_blurred_f = ndimage.gaussian_filter(
            blurred_f, second_filter)
        rst_img = blurred_f + alpha * (blurred_f - filter_blurred_f)
        para_desc = "(%d_%d_%d)" % (
            first_filter, second_filter, alpha)
        return rst_img, para_desc

    def _scale(self, image, params):
        x = float(params['x'])
        y = float(params['y'])
        rst_img = cv2.resize(image, None, fx=x, fy=y,
                             interpolation=cv2.INTER_CUBIC)
        para_desc = "(rescale-X_%s_rescale-Y_%s)" % (x, y)
        return rst_img, para_desc

    def _bilateral_blur(self, image, params):
        diameter = int(params['diameter'])
        sigma_color = int(params['sigma_color'])
        sigma_space = int(params['sigma_space'])

        rst_img = cv2.bilateralFilter(
            image, diameter, sigma_color, sigma_space)
        para_desc = "(%d_%d_%d)" % (
            diameter, sigma_color, sigma_space)
        return rst_img, para_desc

    def _median_blur(self, image, params):
        median_blur_strength = int(params['median_blur_strength'])
        rst_img = cv2.medianBlur(image, median_blur_strength)
        para_desc = "(%d)" % (median_blur_strength)

        return rst_img, para_desc

    def _denoising(self, image, params):
        filterLength = int(params['filter_length'])
        colorComponent = int(params['color_component'])
        tempWindow = int(params['temp_window_size'])
        searchWindow = int(params['search_window_size'])
        rst_img = cv2.fastNlMeansDenoisingColored(image, None, filterLength, colorComponent, tempWindow,
                                                  searchWindow)
        para_desc = "(%d_%d_%d_%d)" % (
            filterLength, colorComponent, tempWindow, searchWindow)
            
        return rst_img, para_desc
    
    def _guassian_blur(self, image, params):
        width = int(params['gamma_width'])
        height = int(params['gamma_height'])
        guassian_blur = int(params['gamma_blur'])
        rst_img = cv2.GaussianBlur(image, (width, height), guassian_blur)
        para_desc = "(%s_%s_%s)" % (width, height, guassian_blur)

        return rst_img, para_desc

    def _horizontal_flipping(self, image, params):
        new_img = cv2.flip(image, 0)
        return new_img, ''

    def _vertical_flipping(self, image, params):
        new_img = cv2.flip(image, 1)
        return new_img, ''

    def _histogram_equalization(self, img, params):
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        cv2.equalizeHist(channels[0], channels[0])
        cv2.merge(channels, ycrcb)
        cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
        return img, ''

    def _rotation(self, img, param):
        cut_edge = False
        angle = param
        if cut_edge == False:
            angle = 360 - angle
            rotated_img = imutils.rotate_bound(img, angle)
        else:
            rotated_img = imutils.rotate(img, angle)

        para_desc = "(%d_angle)" % angle
        return rotated_img, para_desc


    def _get_rotation_ann(self, img_width, img_height, angle=45, point=None, cut_edge=False):
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
        cornerPoint = [[0, 0], [img_width, 0], [
            img_width, img_height], [0, img_height]]
        newpoint = []
        newCornerPoint = []
        for item in point:
            x1 = item[0]
            y1 = item[1]
            x = (x1 - x2) * math.cos(pi / 180.0 * angle) - \
                (y1 - y2) * math.sin(pi / 180.0 * angle) + x2
            y = (x1 - x2) * math.sin(pi / 180.0 * angle) + \
                (y1 - y2) * math.cos(pi / 180.0 * angle) + y2
            newpoint.append([x, y])

        for item in cornerPoint:
            x1 = item[0]
            y1 = item[1]
            x = (x1 - x2) * math.cos(pi / 180.0 * angle) - \
                (y1 - y2) * math.sin(pi / 180.0 * angle) + x2
            y = (x1 - x2) * math.sin(pi / 180.0 * angle) + \
                (y1 - y2) * math.cos(pi / 180.0 * angle) + y2
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

        newXmin = max(int(
            min(newpoint[0][0], newpoint[1][0], newpoint[2][0], newpoint[3][0])) + moreX, 0)
        newXmax = min(int(
            max(newpoint[0][0], newpoint[1][0], newpoint[2][0], newpoint[3][0])) + moreX, nx)
        newYmin = max(int(
            min(newpoint[0][1], newpoint[1][1], newpoint[2][1], newpoint[3][1])) + moreY, 0)
        newYmax = min(int(
            max(newpoint[0][1], newpoint[1][1], newpoint[2][1], newpoint[3][1])) + moreY, ny)
        return ((img_width, img_height),
                (newXmin, newYmin, newXmax, newYmax))

    def _perspective_transform(self, img, param):
        p, flag = param['p'], param['flag']
        rows, cols, ch = img.shape
        pts1 = np.float32([[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]])

        if flag == 1:
            view = np.float32([[-cols * p, 0], [cols * (1 + p), 0], [474, rows], [0, rows]])
        elif flag == 2:
            view = np.float32([[0, 0], [cols, 0], [cols * (1 + p), rows], [-cols * p, rows]])
        elif flag == 3:
            view = np.float32([[0, -rows * p], [cols, 0], [cols, rows], [0, rows * (1 + p)]])
        else:
            view = np.float32([[0, 0], [cols, -rows * p], [cols, rows * (1 + p)], [0, rows]])

        M = cv2.getPerspectiveTransform(pts1, view)
        new_img = cv2.warpPerspective(img, M, (cols, rows))
        para_desc = "(%3.3f_mode:%d)" % (p, flag)
        return new_img, para_desc

    def _print_params(self):
        for key, value in self._params.items():
            logging.info("--- %s" % (key.upper()))