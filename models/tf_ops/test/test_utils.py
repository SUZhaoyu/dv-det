import json

import numpy as np
from point_viz.converter import PointvizConverter
from tqdm import tqdm

from data.utils.normalization import convert_threejs_coors, convert_threejs_bbox

Converter = PointvizConverter("/home/tan/tony/threejs")


def get_rgbs_from_coors(coors):
    norm_coors = coors - np.min(coors, axis=0, keepdims=True)
    norm_coors = norm_coors / np.max(norm_coors, axis=0, keepdims=True)
    return norm_coors * 255.

def fetch_instance(input_list, num_list, id=0):
    accu_num_list = np.cumsum(num_list)
    if id == 0:
        return input_list[:num_list[0], ...]
    else:
        return input_list[accu_num_list[id - 1]:accu_num_list[id], ...]


def plot_points(coors, intensity=None, rgb=None, name='test'):
    Converter.compile(coors=convert_threejs_coors(coors),
                      intensity=intensity,
                      default_rgb=rgb,
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


def plot_points_from_voxels_with_color(voxels, center_coors, resolution, kernel_size=3, mask=-1, name='test'):
    output_coors = []
    output_rgb = []
    half_kernel_size = (kernel_size - 1) / 2
    for i in tqdm(range(len(voxels))):
        r, g, b = np.random.randint(low=0, high=255, size=3)
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
                output_rgb.append([r, g, b])

    output_coors, output_rgb = np.array(output_coors), np.array(output_rgb)
    Converter.compile(coors=convert_threejs_coors(output_coors),
                      default_rgb=output_rgb,
                      task_name=name)

def plot_points_from_roi_voxels(voxels, roi_attrs, kernel_size=5, mask=-1, name='test'):
    output_coors = []
    output_rgb = []
    half_kernel_size = (kernel_size - 1) / 2
    assert voxels.shape[-1] == 3
    for i in tqdm(range(len(voxels))):
        roi_w, roi_l, roi_h, roi_x, roi_y, roi_z, roi_r = roi_attrs[i, :]
        for n in range(kernel_size ** 3):
            r, g, b = voxels[i, n, :]
            if r + g + b > 0:
                x = n % kernel_size
                z = n // (kernel_size ** 2)
                y = (n - z * kernel_size ** 2) // kernel_size
                x_coor = (x - half_kernel_size) * roi_w / kernel_size
                y_coor = (y - half_kernel_size) * roi_l / kernel_size
                z_coor = (z - half_kernel_size) * roi_h / kernel_size + roi_z

                x_coor_r = x_coor * np.cos(roi_r) - y_coor * np.sin(roi_r) + roi_x
                y_coor_r = x_coor * np.sin(roi_r) + y_coor * np.cos(roi_r) + roi_y

                output_coors.append([x_coor_r, y_coor_r, z_coor])
                output_rgb.append([r, g, b])

    output_coors, output_rgb = np.array(output_coors), np.array(output_rgb)
    Converter.compile(coors=convert_threejs_coors(output_coors),
                      bbox_params=convert_threejs_bbox(roi_attrs),
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
