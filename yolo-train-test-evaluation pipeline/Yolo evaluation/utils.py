# coding: utf-8

import numpy as np
import tensorflow as tf
import cv2


def sigmoid(arr):
    """
    sigmoid every element
    :param arr: array with any shape
    :return: array after sigmoid
    """
    arr = np.array(arr, dtype=np.float128)
    return 1.0 / (1.0 + np.exp(-1.0 * arr))


def softmax(arr):
    """
    :param arr: array
    :return: array after softmax
    """
    arr = np.array(arr, dtype=np.float128)
    arr_exp = np.exp(arr)
    return arr_exp / np.expand_dims(np.sum(arr_exp, axis=-1), axis=-1)


def iou_calc1(boxes1, boxes2):
    """
    :param boxes1/2: (xmin, ymin, xmax, ymax)
    :return: IoU
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU


def iou_calc2(boxes1, boxes2):
    """
    :param boxes1/2: (x,y,w,h)，(x,y)is the center of bbox
    :return: IoU
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU


def iou_calc3(boxes1, boxes2):
    """
    :param boxes1/2: Tensor (xmin, ymin, xmax, ymax)
    :return: IoU
    """
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU

def iou_calc4(boxes1, boxes2):
    """
    :param boxes1/2: (x, y, w, h)
    :return: IoU
    """
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU

def GIOU(boxes1, boxes2):
    """
    :param boxes1/2: Tensor (xmin, ymin, xmax, ymax)
    :return: GIoU
    """
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    intersection_left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    intersection_right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    intersection = tf.maximum(intersection_right_down - intersection_left_up, 0.0)
    inter_area = intersection[..., 0] * intersection[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    GIOU = IOU - 1.0 * (enclose_area - union_area) / enclose_area

    return GIOU

def nms(bboxes, score_threshold, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes:
    Assume 6 bboxes which every one of the scores is bigger score_threshold，then bboxes'shape are (N, 6)->(xmin, ymin, xmax, ymax, score, class)
    (xmin, ymin, xmax, ymax) is relative to original image，score = conf * prob，class is bbox's id.
    :return: best_bboxes
    best_bboxes(xmin, ymin, xmax, ymax, score, class)
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = iou_calc1(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            assert method in ['nms', 'soft-nms']
            weight = np.ones((len(iou),), dtype=np.float32)
            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > score_threshold
            cls_bboxes = cls_bboxes[score_mask]
    return best_bboxes


def img_preprocess1(image, bboxes, target_shape, correct_box=True):
    
    h_target, w_target = target_shape
    h_org, w_org, _ = image.shape

    image = np.copy(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (w_target, h_target))
    image = image / 255.0

    if correct_box:
        h_ratio = 1.0 * h_target / h_org
        w_ratio = 1.0 * w_target / w_org
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * w_ratio
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * h_ratio
        return image, bboxes
    return image


def img_preprocess2(image, bboxes, target_shape, correct_box=True):
    
    h_target, w_target = target_shape
    h_org, w_org, _ = image.shape

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    resize_ratio = min(1.0 * w_target / w_org, 1.0 * h_target / h_org)
    resize_w = int(resize_ratio * w_org)
    resize_h = int(resize_ratio * h_org)
    image_resized = cv2.resize(image, (resize_w, resize_h))

    image_paded = np.full((h_target, w_target, 3), 128.0)
    dw = int((w_target - resize_w) / 2)
    dh = int((h_target - resize_h) / 2)
    image_paded[dh:resize_h+dh, dw:resize_w+dw,:] = image_resized
    image = image_paded / 255.0

    if correct_box:
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh
        return image, bboxes
    return image