import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data.kitti_generator import Dataset


def grid_sampling_numpy_batch(input_coors,
                        input_num_list,
                        resolution,
                        dimension=[70.4, 80.0, 4.0],
                        offset=[0., 40.0, 3.0]):
    npoint = input_coors.shape[0]
    batch_size = len(input_num_list)
    dim_w = np.floor(dimension[0] / resolution).astype(np.int64)
    dim_l = np.floor(dimension[1] / resolution).astype(np.int64)
    dim_h = np.floor(dimension[2] / resolution).astype(np.int64)

    input_accu_list = np.cumsum(input_num_list)
    output_coors, output_num_list = [], []
    for b in range(batch_size):
        coors = input_coors[input_accu_list[b]:input_accu_list[b]+input_num_list[b], :]
        voxel_coors = np.floor((coors + offset) / resolution).astype(np.int64)
        voxel_ids = voxel_coors[:, 2] * dim_l * dim_w + voxel_coors[:, 1] * dim_w + voxel_coors[:, 0]
        _, unique_point_idx = np.unique(voxel_ids, return_index=True)
        output_coors.append(coors[unique_point_idx, :])
        output_num_list.append(len(unique_point_idx))

    output_coors = np.concatenate(output_coors, axis=0)
    output_num_list = np.array(output_num_list)

    return output_coors, output_num_list


def grid_sampling_numpy(input_coors,
                        input_num_list,
                        resolution,
                        dimension=[100, 160.0, 8.0],
                        offset=[10., 80.0, 4.0]):
    npoint = input_coors.shape[0]
    batch_size = len(input_num_list)
    dim_w = np.floor(dimension[0] / resolution).astype(np.int64)
    dim_l = np.floor(dimension[1] / resolution).astype(np.int64)
    dim_h = np.floor(dimension[2] / resolution).astype(np.int64)
    dim_offset = dim_w * dim_l * dim_h

    point_ids = np.arange(npoint) + 1
    point_ids_array = np.tile(np.expand_dims(point_ids, axis=0), [batch_size, 1]).astype(np.float32)
    accu_num_list = np.cumsum(input_num_list).astype(np.float32)
    masks = np.greater(point_ids_array / np.expand_dims(accu_num_list, axis=-1), 1.0).astype(np.int64)
    voxel_offset_masks = np.sum(masks, axis=0) * dim_offset

    input_voxel_coors = np.floor((input_coors + offset) / resolution).astype(np.int64)
    input_voxel_ids = input_voxel_coors[:, 2] * dim_l * dim_w + input_voxel_coors[:, 1] * dim_w + input_voxel_coors[:, 0]
    input_voxel_ids += voxel_offset_masks

    _, unique_point_ids = np.unique(input_voxel_ids, return_index=True)
    unique_coors = input_coors[unique_point_ids, :]
    unique_voxels = input_voxel_ids[unique_point_ids]

    voxel_batch_id = np.floor(unique_voxels / dim_offset).astype(np.int32)
    batch_array = np.tile(np.expand_dims(np.arange(batch_size), 1), [1, len(unique_voxels)]).astype(np.int32)
    output_num_list = np.sum(np.equal(voxel_batch_id, batch_array), axis=-1)
    if np.min(voxel_batch_id) < 0 or np.max(voxel_batch_id) > 15:
        print("Warning: voxel_batch_id exceeds range: [{}, {}]".format(np.min(voxel_batch_id), np.max(voxel_batch_id)))

    if (len(unique_coors) != np.sum(output_num_list)):
        print(" ")

    return unique_coors, output_num_list



os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 16
epoch = 1000
if __name__ == '__main__':
    Dataset = Dataset(task='training',
                      batch_size=batch_size,
                      num_worker=6,
                      hvd_size=1,
                      hvd_id=0)
    input_coors, input_features, input_num_list = [], [], []
    for i in tqdm(range(epoch)):
        coors_d, _, num_list_d, _ = next(Dataset.train_generator())
        accu = 0
        coors, num_list = coors_d, num_list_d
        coors, num_list = grid_sampling_numpy(coors, num_list, 0.4)
        coors, num_list = grid_sampling_numpy(coors, num_list, 0.6)
        coors, num_list = grid_sampling_numpy(coors, num_list, 0.8)
        if len(coors) != np.sum(num_list):
            print(len(coors), np.sum(num_list))

