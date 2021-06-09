import json
import tensorflow as tf
import numpy as np
from point_viz.converter import PointvizConverter
from tqdm import tqdm
from copy import deepcopy

from data.utils.normalization import convert_threejs_coors, convert_threejs_bbox, convert_threejs_bbox_with_prob
import time
Converter = PointvizConverter("/home/tan/tony/threejs")


def get_rgbs_from_coors(coors, repeat=5):
    norm_coors = coors - np.min(coors, axis=0, keepdims=True)
    norm_coors = norm_coors / np.max(norm_coors, axis=0, keepdims=True)
    return norm_coors * repeat * 255 % 255.


def get_rgbs_from_coors_tf(coors, repeat=5):
    norm_coors = coors - tf.reduce_min(coors, axis=0, keepdims=True)
    norm_coors = norm_coors / tf.reduce_max(norm_coors, axis=0, keepdims=True)
    return norm_coors * repeat * 255 % 255.


def fetch_instance(input_list, num_list, id=0):
    accu_num_list = np.cumsum(deepcopy(num_list))
    if id == 0:
        return input_list[:num_list[0], ...]
    else:
        return input_list[accu_num_list[id - 1]:accu_num_list[id], ...]


def plot_points(coors, intensity=None, rgb=None, bboxes=None, prob=None, name='test'):
    bbox_params = None if bboxes is None else convert_threejs_bbox_with_prob(bboxes, color=prob)
    Converter.compile(coors=convert_threejs_coors(coors),
                      intensity=intensity,
                      default_rgb=rgb,
                      bbox_params=bbox_params,
                      task_name=name)


def plot_points_from_voxels(voxels, center_coors, resolution, kernel_size=3, mask=-1, name='test'):
    output_coors = []
    output_intensity = []
    half_kernel_size = (kernel_size - 1) / 2
    for i in tqdm(range(len(voxels))):
        for n in range(kernel_size ** 3):
            intensity = voxels[i, n, 0]
            if intensity != mask:
                x = n % kernel_size
                z = n // (kernel_size ** 2)
                y = (n - z * kernel_size ** 2) // kernel_size
                x_coor = (x - half_kernel_size) * resolution + center_coors[i, 0]
                y_coor = (y - half_kernel_size) * resolution + center_coors[i, 1]
                z_coor = (z - half_kernel_size) * resolution + center_coors[i, 2]
                output_coors.append([x_coor, y_coor, z_coor])
                output_intensity.append(intensity)

    output_coors, output_intensity = np.array(output_coors), np.array(output_intensity)
    Converter.compile(coors=convert_threejs_coors(output_coors),
                      intensity=output_intensity,
                      task_name=name)


def plot_points_from_voxels_with_color(voxels, center_coors, resolution, self_rgbs=False, kernel_size=3, mask=-1, name='test'):
    output_coors = []
    output_rgb = []
    half_kernel_size = (kernel_size - 1) / 2
    for i in tqdm(range(len(voxels))):
        r, g, b = np.random.randint(low=0, high=255, size=3)
        for n in range(kernel_size ** 3):
            intensity = voxels[i, n, 0]
            if intensity != mask:
                z = n % kernel_size
                x = n // (kernel_size ** 2)
                y = (n - x * kernel_size ** 2) // kernel_size
                x_coor = (x - half_kernel_size) * resolution + center_coors[i, 0]
                y_coor = (y - half_kernel_size) * resolution + center_coors[i, 1]
                z_coor = (z - half_kernel_size) * resolution + center_coors[i, 2]
                output_coors.append([x_coor, y_coor, z_coor])
                if not self_rgbs:
                    output_rgb.append([r, g, b])
                else:
                    output_rgb.append(voxels[i, n, :])

    output_coors, output_rgb = np.array(output_coors), np.array(output_rgb)
    Converter.compile(coors=convert_threejs_coors(output_coors),
                      default_rgb=output_rgb,
                      task_name=name)



def plot_points_from_roi_voxels(voxels, roi_attrs, kernel_size=5, mask=-1, name='test'):
    output_coors = []
    output_rgb = []
    half_kernel_size = (kernel_size - kernel_size % 2) / 2
    assert voxels.shape[-1] == 3
    for i in tqdm(range(len(voxels))):
        roi_w, roi_l, roi_h, roi_x, roi_y, roi_z, roi_r = roi_attrs[i, :]
        for n in range(kernel_size ** 3):
            r, g, b = voxels[i, n, :]
            if r + g + b > 1e-3:
                x = n // (kernel_size ** 2)
                z = n % kernel_size
                # kernel_id % (kernel_size * kernel_size) / kernel_size;
                y = (n % kernel_size ** 2) // kernel_size
                x_coor = (x - half_kernel_size + 0.5 * (1 - kernel_size % 2)) * roi_w / kernel_size
                y_coor = (y - half_kernel_size + 0.5 * (1 - kernel_size % 2)) * roi_l / kernel_size
                z_coor = (z - half_kernel_size + 0.5 * (1 - kernel_size % 2)) * roi_h / kernel_size + roi_z

                x_coor_r = x_coor * np.cos(roi_r) - y_coor * np.sin(roi_r) + roi_x
                y_coor_r = x_coor * np.sin(roi_r) + y_coor * np.cos(roi_r) + roi_y

                output_coors.append([x_coor_r, y_coor_r, z_coor])
                output_rgb.append([r, g, b])

    output_coors, output_rgb = np.array(output_coors), np.array(output_rgb)
    Converter.compile(coors=convert_threejs_coors(deepcopy(output_coors)),
                      bbox_params=convert_threejs_bbox(deepcopy(roi_attrs)),
                      default_rgb=output_rgb,
                      task_name=name)


class TimeLiner:
    _timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)

def get_points_from_dense_voxels(voxels, resolution, offset, mask=0):
    offset = np.array(offset)
    output_coors = []
    output_features = []
    for w in tqdm(range(voxels.shape[0])):
        for l in range(voxels.shape[1]):
            for h in range(voxels.shape[2]):
                if voxels[w, l, h, 0] > mask:
                    coors = np.array([w*resolution[0], l*resolution[1], h*resolution[2]]) - offset
                    features = voxels[w, l, h, 0]
                    output_coors.append(coors)
                    output_features.append(features)
    output_coors = np.array(output_coors)
    output_features = np.array(output_features)

    return output_coors, output_features





