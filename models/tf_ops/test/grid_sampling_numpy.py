import os

import numpy as np
from tqdm import tqdm

from data.kitti_generator import Dataset
from numpy import floor
from models.tf_ops.test.test_utils import fetch_instance, plot_points

dimension = [140., 140., 9.]
offset = [50., 50., 5.]

def grid_sampling_npy(input_coors, dimension, resolution, base_resolution):
    idx = []
    scale = int(floor(resolution / base_resolution))
    print(scale)
    grid_w = int(int(floor(dimension[0] / base_resolution)) / scale)
    grid_l = int(int(floor(dimension[1] / base_resolution)) / scale)
    grid_h = int(int(floor(dimension[2] / base_resolution)) / scale)
    grid_buffer = np.zeros(shape=[grid_w, grid_l, grid_h], dtype=np.int32)
    for i in range(len(input_coors)):
        grid_base_coor_w = int(floor(input_coors[i, 0] / base_resolution))
        grid_base_coor_l = int(floor(input_coors[i, 1] / base_resolution))
        grid_base_coor_h = int(floor(input_coors[i, 2] / base_resolution))

        grid_coor_w = int(floor(grid_base_coor_w / scale))
        grid_coor_l = int(floor(grid_base_coor_l / scale))
        grid_coor_h = int(floor(grid_base_coor_h / scale))

        # print(i, grid_coor_w, grid_coor_l, grid_coor_h)

        grid_offset_w = grid_base_coor_w % scale
        grid_offset_l = grid_base_coor_l % scale
        grid_offset_h = grid_base_coor_h % scale

        if (grid_offset_w == grid_offset_l == grid_offset_h == 0):
            if (grid_buffer[grid_coor_w, grid_coor_l, grid_coor_h] == 0):
                idx.append(i)
                grid_buffer[grid_coor_w, grid_coor_l, grid_coor_h] = 1
        # else:
        #     idx.append(i)
        #     grid_buffer[grid_coor_w, grid_coor_l, grid_coor_h] = grid_offset_l + 1

    return input_coors[idx, :], grid_buffer

def buffer_to_coors(buffer, dimension):
    output_coors = []
    output_features = []
    w, l, h = buffer.shape
    maximum = np.max(buffer)
    print(maximum)
    for x in tqdm(range(w)):
        for y in range(l):
            for z in range(h):
                if buffer[x, y, z] > 0:
                    x_coor = x * dimension[0] / w
                    y_coor = y * dimension[1] / l
                    z_coor = z * dimension[2] / h
                    output_coors.append([x_coor, y_coor, z_coor])
                    output_features.append(buffer[x, y, z] / maximum)
    return np.array(output_coors), np.array(output_features)


if __name__ == '__main__':
    Dataset = Dataset(task='validation',
                      batch_size=1,
                      num_worker=1,
                      validation=True,
                      hvd_size=1,
                      hvd_id=0)
    coors_d, features_d, num_list_d, _ = next(Dataset.valid_generator(start_idx=71))

    coors, buffer = grid_sampling_npy(input_coors=coors_d + offset,
                                      dimension=dimension,
                                      resolution=0.1,
                                      base_resolution=0.1)

    # coors, features = buffer_to_coors(buffer, dimension)
    # plot_points(coors=coors-offset, intensity=features, name='grid_sampling')

    plot_points(coors=coors-offset, name='grid_sampling')
