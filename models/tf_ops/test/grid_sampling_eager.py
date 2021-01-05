import os

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from tqdm import tqdm

from data.kitti_generator import Dataset
tf.enable_eager_execution()
from models.tf_ops.custom_ops import grid_sampling
from models.tf_ops.test.test_utils import fetch_instance, plot_points

unique_exe = tf.load_op_library('/home/tan/tony/dv-det/models/tf_ops/build/unique.so')
def grid_sampling_thrust(input_coors,
                        input_num_list,
                        resolution,
                        dimension=[70.4, 80.0, 4.0],
                        offset=[0., 40.0, 3.0]):
    '''
    The grid sub-sampling strategy, aims at taking the place of FPS. This operation intents to yield uniformly distributed
sampling result. This operation is implemented in stack style, which means the number of points of each input instance
    does not have to be fixed number, and difference instances are differentiated using "input_num_list".

    :param input_coors: 2-D tf.float32 Tensor with shape=[input_npoint, channels].
    :param input_num_list: 1-D tf.int32 Tensor with shape=[batch_size], indicating how many points within each instance.
    :param resolution: float32, the down sampleing resolution.
    :param dimension: 1-D float32 list with shape 3, the maximum in x, y, z orientation of the input coors, this will be used to
                      create the unique voxel ids for each input points
    :param offset: 1-D float32 list with shape 3, the offset on each axis, so that the minimum coors in each axis is > 0.
    :return:
    output_coors: 2-D tf.float32 Tensor with shape=[output_npoint, channels], the output coordinates of the sub-sampling.
    output_num_list: 1-D tf.int32 Tensor with shape=[batch_size], same definition as input_num_list.
    '''

    npoint = tf.shape(input_coors)[0]
    batch_size = tf.shape(input_num_list)[0]
    dim_w = tf.cast(tf.ceil(dimension[0] / resolution), dtype=tf.int64)
    dim_l = tf.cast(tf.ceil(dimension[1] / resolution), dtype=tf.int64)
    dim_h = tf.cast(tf.ceil(dimension[2] / resolution), dtype=tf.int64)
    dim_offset = dim_w * dim_l * dim_h

    point_ids = tf.range(npoint) + 1
    point_ids_array = tf.cast(tf.tile(tf.expand_dims(point_ids, axis=0), [batch_size, 1]), dtype=tf.float32)
    accu_num_list = tf.cast(tf.cumsum(input_num_list), dtype=tf.float32)
    masks = tf.cast(tf.greater(point_ids_array / tf.expand_dims(accu_num_list, axis=-1), 1.0), dtype=tf.int64)
    voxel_offset_masks = tf.reduce_sum(masks, axis=0) * dim_offset

    input_voxel_coors = tf.cast(tf.floor((input_coors + offset) / resolution), dtype=tf.int64)
    input_voxel_ids = input_voxel_coors[:, 2] * dim_l * dim_w + input_voxel_coors[:, 1] * dim_w + input_voxel_coors[:, 0]
    input_voxel_ids += voxel_offset_masks
    input_point_ids = tf.range(npoint, dtype=tf.int32)

    unique_point_ids = unique_exe.unique_op(input_voxel_ids=input_voxel_ids,
                                            input_point_ids=input_point_ids)

    unique_coors = tf.gather(input_coors, unique_point_ids, axis=0)
    unique_voxels = tf.gather(input_voxel_ids, unique_point_ids, axis=0)

    voxel_batch_id = tf.cast(tf.floor(unique_voxels / dim_offset), dtype=tf.int32)
    batch_array = tf.cast(tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, tf.shape(unique_voxels)[0]]), dtype=tf.int32)
    output_num_list = tf.reduce_sum(tf.cast(tf.equal(voxel_batch_id, batch_array), dtype=tf.int32), axis=-1)


    # unique_voxels_array = tf.cast(tf.tile(tf.expand_dims(unique_voxels, 0), [batch_size, 1]), dtype=tf.float32)
    # bottom_offset_list = tf.cast(tf.range(batch_size), dtype=tf.float32) * tf.to_float(dim_offset)
    # upper_offset_list = tf.cast(tf.range(batch_size) + 1, dtype=tf.float32) * tf.to_float(dim_offset)
    # bottom_masks = tf.cast(tf.greater(unique_voxels_array / tf.expand_dims(bottom_offset_list, axis=-1), 1.0), dtype=tf.float32)
    # up_masks = tf.cast(tf.greater(unique_voxels_array / tf.expand_dims(upper_offset_list, axis=-1), 1.0), dtype=tf.float32)
    # bottom_count = tf.reduce_sum(bottom_masks, axis=-1)
    # up_count = tf.reduce_sum(up_masks, axis=-1)
    # output_num_list = tf.cast(bottom_count - up_count, tf.int32)

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
        coors, num_list = grid_sampling_thrust(coors, num_list, 0.4)
        coors, num_list = grid_sampling_thrust(coors, num_list, 0.6)
        coors, num_list = grid_sampling_thrust(coors, num_list, 0.8)
        if np.array(coors.shape[0]) != np.array(tf.reduce_sum(num_list)):
            print(np.array(coors.shape[0]), np.array(tf.reduce_sum(num_list)))

