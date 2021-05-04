from __future__ import division

import math
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def normalize_angle(angle):
    """  Returns the angle (in radians) between -pi and +pi that corresponds to the input angle

        Args:
           * **angle** (float):  Angle to normalize

        Returns:
           float  Angle

    """
    angle = angle % (2 * np.pi)
    angle -= 2 * np.pi * np.array(angle > np.pi, dtype=np.float32)
    return angle

def get_union_sets(conditions):
    output = conditions[0]
    for i in np.arange(1, len(conditions)):
        output = np.logical_and(output, conditions[i])
    return output

def range_clip(coors, range_x, range_y, range_z, offset=0.1):
    assert len(coors.shape) == 2 and coors.shape[-1] == 3
    keep_idx = get_union_sets([coors[:, 0] > range_x[0] + offset,
                               coors[:, 0] < range_x[1] - offset,
                               coors[:, 1] > range_y[0] + offset,
                               coors[:, 1] < range_y[1] - offset,
                               coors[:, 2] > range_z[0] + offset,
                               coors[:, 2] < range_z[1] - offset])
    return keep_idx

def length_normalize(points, length, warning=False):
    assert len(points.shape) == 2
    actual_length = len(points)
    if warning and actual_length != length:
        print("WARNING: the received data length does not match the expected length.")
    if actual_length > length:
        selected_idxs = np.random.choice(range(actual_length), length, replace=False)
        return points[selected_idxs]
    elif actual_length < length:
        padding_length = length - actual_length
        padding_idxs = np.random.choice(range(actual_length), padding_length)
        padding_points = points[padding_idxs]
        points = np.concatenate([points, padding_points], axis=0)
    return points


def feature_normalize(features, method):
    assert len(features.shape) == 2
    if method == 'L2':
        m = np.expand_dims(np.sqrt(np.sum(features ** 2, axis=1)), axis=-1)
        features /= m
    elif method == '-1~1':
        features -= np.min(features)
        features /= np.max(features)
        features = (features - 0.5) * 2.
    elif method == '0~1':
        features -= np.min(features)
        features /= np.max(features)
    elif method == 'max_1':
        m = np.expand_dims(np.max(np.abs(features), axis=1), axis=-1)
        features /= m
    elif method == '255':
        features /= 255.0
    elif method == 'channel_std':
        if len(features) > 0:
            features -= np.mean(features, axis=0)
            features = features / (np.std(features, axis=0) + 1e-6)
    elif method == 'global_std':
        features -= np.mean(features)
        features /= np.std(features)
    elif method == None:
        features = features
    else:
        raise ValueError("Unsupported normalization method: {}".format(method))
    return features


def image_feature_normalize(features, method):
    assert len(features.shape) == 3
    if method == 'channel_std':
        features -= np.mean(features, axis=(0, 1))
        features /= np.std(features, axis=(0, 1))
    else:
        raise ValueError("Unsupported normalization method: {}".format(method))
    return features


def coor_normalize(coors):
    assert len(coors.shape) == 2
    coors_min = np.min(coors, axis=0)
    coors_max = np.max(coors, axis=0)
    coors_center = (coors_min + coors_max) / 2.
    coors -= coors_center
    m = np.max(np.abs(coors))
    coors /= m
    return coors


def convert_threejs_coors(coors):
    assert len(coors.shape) == 2
    threejs_coors = np.zeros(shape=coors.shape)
    threejs_coors[:, 0] = coors[:, 1]
    threejs_coors[:, 1] = coors[:, 2]
    threejs_coors[:, 2] = coors[:, 0]
    return threejs_coors


def convert_threejs_bbox(bboxes):
    assert len(bboxes) > 0
    for bbox in bboxes:
        w, l, h, x, y, z = bbox[:6]
        bbox[:6] = [l, h, w, y, z, x]

    threejs_bboxes = []
    category_dict = ["Car", "Van", "Pedestrian", "Cyclist"]
    difficulty_dict = ["Easy", "Moderate", "Hard", "Unknown", "Omitted"]
    for box in bboxes:
        if box[0] * box[1] * box[2] > 0:
            threejs_bbox = [0] * 9
            threejs_bbox[:7] = box[:7]
            # threejs_bbox[-1] = category_dict[int(box[-2])] + ', ' + difficulty_dict[int(box[-1])]
            threejs_bbox[-1] = "%.2fpi,%d" % (box[6]/np.pi, box[-1])
            threejs_bbox[-2] = "Magenta"
            threejs_bboxes.append(threejs_bbox)

    return threejs_bboxes


def bboxes_normalization(bboxes, length=64, diff_thres=None):
    bbox_attr_num = 9
    if diff_thres is not None:
        for i in range(len(bboxes)):
            if int(bboxes[i][-1]) > diff_thres:
                bboxes[i] = [0] * bbox_attr_num
    bboxes = np.array(bboxes)
    if bboxes.shape[0] > length:
        print("WARNING: number of bboxes {} exceeds the buffer limition: {}".format(bboxes.shape[0], length))
    padding_bboxes = np.zeros((length, bbox_attr_num))
    # padding_bboxes[:, :3] = np.ones_like(padding_bboxes[:, :3]) * 1e-2
    if len(bboxes) != 0:
        padding_bboxes[:len(bboxes), ...] = bboxes
    return padding_bboxes


def convert_threejs_bbox_with_id(bboxes, color="white"):
    assert len(bboxes) > 0
    for bbox in bboxes:
        w, l, h, x, y, z = bbox[:6]
        bbox[:6] = [l, h, w, y, z, x]

    threejs_bboxes = []
    category_dict = ["Car", "Pedestrian", "Cyclist", ""]
    difficulty_dict = ["Easy", "Moderate", "Hard", "Unknown", "Omitted", ""]
    for i, box in enumerate(bboxes):
        if box[0] * box[1] * box[2] > 0:
            threejs_bbox = [0] * 9
            threejs_bbox[:7] = box[:7]
            threejs_bbox[-1] = category_dict[int(box[-2])] + ': ' + str(i)
            threejs_bbox[-2] = color
            threejs_bboxes.append(threejs_bbox)
    return threejs_bboxes


def convert_threejs_bbox_with_prob(bboxes, color=None):
    viridis = cm.get_cmap('viridis', lut=10)
    magma = cm.get_cmap('magma')
    hot = cm.get_cmap('hot')
    seismic = cm.get_cmap('seismic')
    bboxes = deepcopy(bboxes)
    assert len(bboxes) > 0
    for bbox in bboxes:
        w, l, h, x, y, z = bbox[:6]
        bbox[:6] = [l, h, w, y, z, x]

    threejs_bboxes = []
    for i, box in enumerate(bboxes):
        if box[0] * box[1] * box[2] > 0:
            threejs_bbox = [0] * 9
            threejs_bbox[:7] = box[:7]
            if color is not None:
                threejs_bbox[-1] = "%0.2f" % color[i]
            else:
                threejs_bbox[-1] = " "
            # threejs_bbox[-1] = "%0.2f" % box[6]
            if color is not None:
                if math.isnan(box[-1]):
                    threejs_bbox[-2] = 'fuchsia'
                else:
                    rgb = viridis(color[i])
                    threejs_bbox[-2] = 'rgb({},{},{})'.format(int(255*rgb[0]), int(255*rgb[1]), int(255*rgb[2]))
            else:
                threejs_bbox[-2] = 'white'
            threejs_bboxes.append(threejs_bbox)

    return threejs_bboxes


def convert_threejs_bbox_with_colors(bboxes, color="white"):
    bboxes = deepcopy(bboxes)
    assert len(bboxes) > 0
    for bbox in bboxes:
        w, l, h, x, y, z = bbox[:6]
        bbox[:6] = [l, h, w, y, z, x]

    threejs_bboxes = []
    for i, box in enumerate(bboxes):
        if box[0] * box[1] * box[2] > 0:
            threejs_bbox = [0] * 9
            threejs_bbox[:7] = box[:7]
            threejs_bbox[-1] = "%d, %0.2f" % (int(box[8]), box[-1])
            # threejs_bbox[-1] = "conf=%0.2f, iou=%0.2f" % (box[-2], box[-1])
            threejs_bbox[-2] = color
            # if (box[-1] == 0):
            threejs_bboxes.append(threejs_bbox)
    return threejs_bboxes


def convert_threejs_bbox_with_assigned_colors(bboxes, colors):
    bboxes = deepcopy(bboxes)
    assert len(bboxes) > 0 and len(bboxes) == len(colors)
    for bbox in bboxes:
        w, l, h, x, y, z = bbox[:6]
        bbox[:6] = [l, h, w, y, z, x]

    threejs_bboxes = []
    for i, box in enumerate(bboxes):
        if box[0] * box[1] * box[2] > 0:
            threejs_bbox = [0] * 9
            threejs_bbox[:7] = box[:7]
            threejs_bbox[-1] = " "
            # threejs_bbox[-1] = "conf=%0.2f, iou=%0.2f" % (box[-2], box[-1])
            threejs_bbox[-2] = colors[i]
            # if (box[-1] == 0):
            threejs_bboxes.append(threejs_bbox)
    return threejs_bboxes


def get_points_rgb_with_prob(conf):
    viridis = cm.get_cmap('viridis', lut=10)
    rgbs = viridis(conf) * 255
    return rgbs[:, :3]