import os
from os.path import join

import tensorflow as tf
from tensorflow.python.framework import ops

CWD = os.path.dirname(os.path.abspath(__file__))

# =============================================Grid Sub Sample===============================================

unique_exe = tf.load_op_library(join(CWD, 'build', 'unique.so'))
def grid_sub_sample(input_coors,
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
    dim_w = tf.cast(tf.floor(dimension[0] / resolution), dtype=tf.int64)
    dim_l = tf.cast(tf.floor(dimension[1] / resolution), dtype=tf.int64)
    dim_h = tf.cast(tf.floor(dimension[2] / resolution), dtype=tf.int64)
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

    unique_voxels_array = tf.cast(tf.tile(tf.expand_dims(unique_voxels, 0), [batch_size, 1]), dtype=tf.float32)
    bottom_offset_list = tf.cast(tf.range(batch_size), dtype=tf.float32) * tf.to_float(dim_offset)
    upper_offset_list = tf.cast(tf.range(batch_size) + 1, dtype=tf.float32) * tf.to_float(dim_offset)
    bottom_masks = tf.cast(tf.greater(unique_voxels_array / tf.expand_dims(bottom_offset_list, axis=-1), 1.0), dtype=tf.float32)
    up_masks = tf.cast(tf.greater(unique_voxels_array / tf.expand_dims(upper_offset_list, axis=-1), 1.0), dtype=tf.float32)
    bottom_count = tf.reduce_sum(bottom_masks, axis=-1)
    up_count = tf.reduce_sum(up_masks, axis=-1)
    output_num_list = bottom_count - up_count

    return unique_coors, tf.cast(output_num_list, tf.int32)

ops.NoGradient("UniqueOp")

# =============================================Dynamic Voxelization===============================================

voxel_sample_exe = tf.load_op_library(join(CWD, 'build', 'voxel_sample.so'))
def dynamic_voxelization(input_coors,
                         input_features,
                         input_num_list,
                         center_coors,
                         center_num_list,
                         resolution,
                         padding=0.,
                         dimension=[70.4, 80.0, 4.0],
                         offset=[0., 40.0, 3.0]):

    npoint = tf.shape(input_coors)[0]
    batch_size = tf.shape(input_num_list)[0]
    dim_w = tf.cast(tf.floor(dimension[0] / resolution), dtype=tf.int64)
    dim_l = tf.cast(tf.floor(dimension[1] / resolution), dtype=tf.int64)
    dim_h = tf.cast(tf.floor(dimension[2] / resolution), dtype=tf.int64)
    dim_offset = dim_w * dim_l * dim_h

    point_ids = tf.range(npoint) + 1
    point_ids_array = tf.cast(tf.tile(tf.expand_dims(point_ids, axis=0), [batch_size, 1]), dtype=tf.float32)
    accu_num_list = tf.cast(tf.cumsum(input_num_list), dtype=tf.float32)
    masks = tf.cast(tf.greater(point_ids_array / tf.expand_dims(accu_num_list, axis=-1), 1.0), dtype=tf.int64)
    voxel_offset_masks = tf.reduce_sum(masks, axis=0) * dim_offset

    input_voxel_coors = tf.cast(tf.floor((input_coors + offset) / resolution), dtype=tf.int64)
    input_voxel_ids = input_voxel_coors[:, 2] * dim_l * dim_w + input_voxel_coors[:, 1] * dim_w + input_voxel_coors[:, 0]
    input_voxel_ids += voxel_offset_masks
    sorted_args = tf.argsort(input_voxel_ids)
    sorted_voxel_ids = tf.gather(input_voxel_ids, sorted_args) - voxel_offset_masks
    sorted_coors = tf.gather(input_coors, sorted_args, axis=0)
    sorted_features = tf.gather(input_features, sorted_args, axis=0)

    output_voxels, _ = voxel_sample_exe.voxel_sample_op(input_coors=sorted_coors + offset,
                                                        input_features=sorted_features,
                                                        input_voxel_idx=sorted_voxel_ids,
                                                        input_num_list=input_num_list,
                                                        center_coors=center_coors + offset,
                                                        center_num_list=center_num_list,
                                                        dimension=dimension,
                                                        resolution=resolution,
                                                        padding_value=padding)
    return output_voxels

@ops.RegisterGradient("VoxelSampleOp")
def voxel_sample_grad(op, grad, _):
    input_features = op.inputs[1]
    output_idx = op.outputs[1]
    input_features_grad = voxel_sample_exe.voxel_sample_grad_op(output_idx=output_idx,
                                                                input_features=input_features,
                                                                output_features_grad=grad)
    return [None, input_features_grad, None, None, None, None]


# =============================================Dense Convolution===============================================

kernel_conv_exe = tf.load_op_library(join(CWD, 'build', 'kernel_conv.so'))
def kernel_conv(input_voxels, filter):
    '''
    Get convolution result between input and filter
    :param input: 3-D Tensor with shape [kernel_number, kernel_size**3, input_channels]
    :param filters: 3-D Tensor with Shape [kernel_size**3, input_channels, output_channels]
    :return:
    output: 2-D Tensor with shape [kernel_number, output_channels]
    '''
    output = kernel_conv_exe.kernel_conv_op(input=input_voxels,
                                          filter=filter)
    return output

@ops.RegisterGradient("KernelConvOp")
def kernel_conv_grad(op, grad):
    input_voxels = op.inputs[0]
    filters = op.inputs[1]
    input_grad, filter_grad = kernel_conv_exe.kernel_conv_grad_op(input=input_voxels,
                                                                 filter=filters,
                                                                 output_grad=grad)
    return [input_grad, filter_grad]