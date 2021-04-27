import os
from os.path import join

import tensorflow as tf
from tensorflow.python.framework import ops

CWD = os.path.dirname(os.path.abspath(__file__))

# =============================================Grid Sampling===============================================

grid_sampling_exe = tf.load_op_library(join(CWD, '../build', 'grid_sampling.so'))

def grid_sampling(input_coors,
                  input_num_list,
                  resolution,
                  dimension,
                  offset):
    output_idx, output_num_list = grid_sampling_exe.grid_sampling_op(input_coors=input_coors + offset,
                                                                     input_num_list=input_num_list,
                                                                     dimension=dimension,
                                                                     resolution=resolution)
    output_coors = tf.gather(input_coors, output_idx, axis=0)
    return output_coors, output_num_list, output_idx

ops.NoGradient("GridSamplingOp")

# =============================================Grid Sampling Thrust===============================================

unique_exe = tf.load_op_library(join(CWD, '../build', 'unique.so'))
def grid_sampling_thrust(input_coors,
                        input_num_list,
                        resolution,
                        dimension,
                        offset):
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
    # input_voxel_coors = tf.clip_by_value(input_voxel_coors, clip_value_min=0, clip_value_max=[dim_w - 1, dim_l - 1, dim_h - 1])
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

    return unique_coors, output_num_list, unique_point_ids

ops.NoGradient("UniqueOp")

# =============================================Voxel Sampling===============================================

voxel_sampling_exe = tf.load_op_library(join(CWD, '../build', 'voxel_sampling.so'))


def voxel_sampling(input_coors,
                   input_features,
                   input_num_list,
                   center_coors,
                   center_num_list,
                   resolution,
                   padding,
                   dimension,
                   offset):
    output_voxels, _ = voxel_sampling_exe.voxel_sampling_op(input_coors=input_coors + offset,
                                                            input_features=input_features,
                                                            input_num_list=input_num_list,
                                                            center_coors=center_coors + offset,
                                                            center_num_list=center_num_list,
                                                            dimension=dimension,
                                                            resolution=resolution,
                                                            padding_value=padding)
    return output_voxels


@ops.RegisterGradient("VoxelSamplingOp")
def voxel_sampling_grad(op, grad, _):
    input_features = op.inputs[1]
    output_idx = op.outputs[1]
    input_features_grad = voxel_sampling_exe.voxel_sampling_grad_op(output_idx=output_idx,
                                                                    input_features=input_features,
                                                                    output_features_grad=grad)
    return [None, input_features_grad, None, None, None]

# =============================================Voxel Sampling Idx===============================================

voxel_sampling_idx_exe = tf.load_op_library(join(CWD, '../build', 'voxel_sampling_idx.so'))


def voxel_sampling_idx(input_coors,
                       input_features,
                       input_num_list,
                       center_coors,
                       center_num_list,
                       resolution,
                       dimension,
                       offset,
                       grid_buffer_size,
                       output_pooling_size,
                       with_rpn=False):
    output_idx, valid_idx = voxel_sampling_idx_exe.voxel_sampling_idx_op(input_coors=input_coors + offset,
                                                                         input_num_list=input_num_list,
                                                                         center_coors=center_coors + offset,
                                                                         center_num_list=center_num_list,
                                                                         dimension=dimension,
                                                                         resolution=resolution,
                                                                         grid_buffer_size=grid_buffer_size,
                                                                         output_pooling_size=output_pooling_size,
                                                                         with_rpn=with_rpn)
    return output_idx, valid_idx, input_features


ops.NoGradient("VoxelSamplingIdxOp")

# =============================================Voxel Sampling Feature===============================================

voxel_sampling_feature_exe = tf.load_op_library(join(CWD, '../build', 'voxel_sampling_feature.so'))


def voxel_sampling_feature(input_features,
                           output_idx,
                           padding):
    output_features = voxel_sampling_feature_exe.voxel_sampling_feature_op(input_features=input_features,
                                                                           output_idx=output_idx,
                                                                           padding_value=padding)
    return output_features

@ops.RegisterGradient("VoxelSamplingFeatureOp")
def voxel_sampling_feature_grad(op, grad):
    input_features = op.inputs[0]
    output_idx = op.inputs[1]
    input_features_grad = voxel_sampling_feature_exe.voxel_sampling_feature_grad_op(input_features=input_features,
                                                                                    output_idx=output_idx,
                                                                                    output_features_grad=grad)
    return [input_features_grad, None]

def voxel_sampling_feature_grad_test(input_features, output_idx, grad):
    input_features_grad = voxel_sampling_feature_exe.voxel_sampling_feature_grad_op(input_features=input_features,
                                                                                    output_idx=output_idx,
                                                                                    output_features_grad=grad)
    return input_features_grad

# =============================================Voxel Sampling Binary===============================================

voxel_sampling_binary_exe = tf.load_op_library(join(CWD, '../build', 'voxel_sampling_binary.so'))
def voxel_sampling_binary(input_coors,
                         input_features,
                         input_num_list,
                         center_coors,
                         center_num_list,
                         resolution,
                         padding,
                         dimension,
                         offset):

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
    # input_voxel_coors = tf.clip_by_value(input_voxel_coors, clip_value_min=0, clip_value_max=[dim_w - 1, dim_l - 1, dim_h - 1])
    input_voxel_ids = input_voxel_coors[:, 2] * dim_l * dim_w + input_voxel_coors[:, 1] * dim_w + input_voxel_coors[:, 0]
    input_voxel_ids += voxel_offset_masks
    sorted_args = tf.argsort(input_voxel_ids)
    sorted_voxel_ids = tf.gather(input_voxel_ids, sorted_args) - voxel_offset_masks
    sorted_coors = tf.gather(input_coors, sorted_args, axis=0)
    sorted_features = tf.gather(input_features, sorted_args, axis=0)
    # XXX: Need to pay attention to the back-propagation implementation.
    output_voxels, _ = voxel_sampling_binary_exe.voxel_sampling_binary_op(input_coors=sorted_coors + offset,
                                                                          input_features=sorted_features,
                                                                          input_voxel_idx=sorted_voxel_ids,
                                                                          input_num_list=input_num_list,
                                                                          center_coors=center_coors + offset,
                                                                          center_num_list=center_num_list,
                                                                          dimension=dimension,
                                                                          resolution=resolution,
                                                                          padding_value=padding)
    return output_voxels

@ops.RegisterGradient("VoxelSamplingBinaryOp")
def voxel_sampling_binary_grad(op, grad, _):
    input_features = op.inputs[1]
    output_idx = op.outputs[1]
    input_features_grad = voxel_sampling_binary_exe.voxel_sampling_binary_grad_op(input_features=input_features,
                                                                                  output_idx=output_idx,
                                                                                  output_features_grad=grad)
    return [None, input_features_grad, None, None, None, None]


# =============================================Voxel Sampling Idx Binary===============================================

voxel_sampling_idx_binary_exe = tf.load_op_library(join(CWD, '../build', 'voxel_sampling_idx_binary.so'))
def voxel_sampling_idx_binary(input_coors,
                              input_features,
                              input_num_list,
                              center_coors,
                              center_num_list,
                              resolution,
                              dimension,
                              offset,
                              grid_buffer_size,
                              output_pooling_size,
                              with_rpn=False):

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
    # input_voxel_coors = tf.clip_by_value(input_voxel_coors, clip_value_min=0, clip_value_max=[dim_w - 1, dim_l - 1, dim_h - 1])
    input_voxel_ids = input_voxel_coors[:, 2] * dim_l * dim_w + input_voxel_coors[:, 1] * dim_w + input_voxel_coors[:, 0]
    input_voxel_ids += voxel_offset_masks
    sorted_args = tf.argsort(input_voxel_ids)
    sorted_voxel_ids = tf.gather(input_voxel_ids, sorted_args) - voxel_offset_masks
    sorted_coors = tf.gather(input_coors, sorted_args, axis=0)
    sorted_features = tf.gather(input_features, sorted_args, axis=0)
    # XXX: Need to pay attention to the back-propagation implementation.
    output_idx, valid_idx = voxel_sampling_idx_binary_exe.voxel_sampling_idx_binary_op(input_coors=sorted_coors + offset,
                                                                                       input_voxel_idx=sorted_voxel_ids,
                                                                                       input_num_list=input_num_list,
                                                                                       center_coors=center_coors + offset,
                                                                                       center_num_list=center_num_list,
                                                                                       dimension=dimension,
                                                                                       resolution=resolution,
                                                                                       grid_buffer_size=grid_buffer_size,
                                                                                       output_pooling_size=output_pooling_size,
                                                                                       with_rpn=with_rpn)
    return output_idx, valid_idx, sorted_features

ops.NoGradient("VoxelSamplingIdxBinaryOp")

