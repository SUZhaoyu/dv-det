import os
from os.path import join

import tensorflow as tf
from tensorflow.python.framework import ops

CWD = os.path.dirname(os.path.abspath(__file__))

# grid_sampling_exe = tf.load_op_library(join(CWD, 'build', 'grid_sampling.so'))
# def grid_sampling(input_coors,
#                   input_num_list,
#                   resolution,
#                   dimension=[100, 160.0, 9.0],
#                   offset=[10., 60.0, 5.0]):
#     output_idx, output_num_list = grid_sampling_exe.grid_sampling_op(input_coors=input_coors + offset,
#                                                                      input_num_list=input_num_list,
#                                                                      dimension=dimension,
#                                                                      resolution=resolution)
#     output_coors = tf.gather(input_coors, output_idx, axis=0)
#     return output_coors, output_num_list, output_idx
#
#
# ops.NoGradient("GridSamplingOp")
#
#
#
# # =============================================Voxel Sampling===============================================
#
# voxel_sampling_exe = tf.load_op_library(join(CWD, 'build', 'voxel_sampling.so'))
#
#
# def voxel_sampling(input_coors,
#                    input_features,
#                    input_num_list,
#                    center_coors,
#                    center_num_list,
#                    resolution,
#                    padding=0.,
#                    dimension=[100, 160.0, 9.0],
#                    offset=[10., 60.0, 5.0]):
#     output_voxels, output_idx = voxel_sampling_exe.voxel_sampling_op(input_coors=input_coors + offset,
#                                                                      input_features=input_features,
#                                                                      input_num_list=input_num_list,
#                                                                      center_coors=center_coors + offset,
#                                                                      center_num_list=center_num_list,
#                                                                      dimension=dimension,
#                                                                      resolution=resolution,
#                                                                      padding_value=padding)
#     return output_voxels, output_idx
#
#
# @ops.RegisterGradient("VoxelSamplingOp")
# def voxel_sampling_grad(op, grad, _):
#     input_features = op.inputs[1]
#     output_idx = op.outputs[1]
#     input_features_grad = voxel_sampling_exe.voxel_sampling_grad_op(output_idx=output_idx,
#                                                                     input_features=input_features,
#                                                                     output_features_grad=grad)
#     return [None, input_features_grad, None, None, None]
#
#
#
#
# # =============================================Get RoI Ground Truth===============================================
#
# get_roi_bbox_exe = tf.load_op_library(join(CWD, 'build', 'get_roi_bbox.so'))
#
#
# def get_roi_bbox(input_coors, bboxes, input_num_list, anchor_size, expand_ratio=0.15, diff_thres=3):
#     '''
#     Get point-wise RoI ground truth.
#     :param input_coors: 2-D Tensor with shape [npoint, 3]
#     :param bboxes: 3-D Tensor with shape [batch, nbbox, bbox_attr]
#     :param input_num_list: 1-D Tensor with shape [batch]
#     :param anchor_size: 1-D list or tensor with shape [4] (w, l, h, z)
#     :param channels: int, how many attributes in the final roi output.
#     :param expand_ratio: default=0.1
#     :param diff_thres: default=3, only the points with difficulty <= diff_thres will be linked to the final loss
#     :return: output_attrs: 2-D Tensor with shape [npoint, 10]
#                            [confidence, w, l, h, offset_x, offset_y, offset_z, angle, *face_direction(binary), *class]
#     '''
#     roi_attrs, roi_conf, roi_diff = get_roi_bbox_exe.get_roi_bbox_op(input_coors=input_coors,
#                                                                       gt_bbox=bboxes,
#                                                                       input_num_list=input_num_list,
#                                                                       anchor_size=anchor_size,
#                                                                       expand_ratio=expand_ratio,
#                                                                       diff_thres=diff_thres)
#     return roi_attrs, roi_conf, roi_diff
#
#
# ops.NoGradient("GetRoiBboxOp")
#
# # =============================================Get Bbox Ground Truth===============================================
#
# get_bbox_exe = tf.load_op_library(join(CWD, 'build', 'get_bbox.so'))
# def get_bbox(roi_attrs, bboxes, input_num_list, expand_ratio=0.15, diff_thres=3):
#     bbox_attrs, bbox_conf, bbox_diff = get_bbox_exe.get_bbox_op(roi_attrs=roi_attrs,
#                                                                 gt_bbox=bboxes,
#                                                                 input_num_list=input_num_list,
#                                                                 expand_ratio=expand_ratio,
#                                                                 diff_thres=diff_thres)
#     return bbox_attrs, bbox_conf, bbox_diff
# ops.NoGradient("GetBboxOp")
#
# # =============================================Roi Pooling===============================================
#
# roi_pooling_exe = tf.load_op_library(join(CWD, 'build', 'roi_pooling.so'))
# def roi_pooling(input_coors, input_features, roi_attrs, input_num_list, roi_num_list,
#                 voxel_size=5, padding_value=0., pooling_size=5):
#     output_features, _ = roi_pooling_exe.roi_pooling_op(input_coors=input_coors,
#                                                         input_features=input_features,
#                                                         roi_attrs=roi_attrs,
#                                                         input_num_list=input_num_list,
#                                                         roi_num_list=roi_num_list,
#                                                         voxel_size=voxel_size,
#                                                         padding_value=padding_value,
#                                                         pooling_size=pooling_size)
#     return output_features
#
# @ops.RegisterGradient("RoiPoolingOp")
# def roi_pooling_grad(op, grad, _):
#     input_features = op.inputs[1]
#     output_idx = op.outputs[1]
#     input_features_grad = roi_pooling_exe.roi_pooling_grad_op(input_features=input_features,
#                                                               output_idx=output_idx,
#                                                               output_features_grad=grad)
#     return [None, input_features_grad, None, None, None]
#
# # =============================================La Roi Pooling===============================================
#
# la_roi_pooling_exe = tf.load_op_library(join(CWD, 'build', 'la_roi_pooling.so'))
# def la_roi_pooling(input_coors, input_features, roi_attrs, input_num_list, roi_num_list,
#                 voxel_size=5, padding_value=0., pooling_size=5):
#     output_features, _, _ = la_roi_pooling_exe.la_roi_pooling_op(input_coors=input_coors,
#                                                                  input_features=input_features,
#                                                                  roi_attrs=roi_attrs,
#                                                                  input_num_list=input_num_list,
#                                                                  roi_num_list=roi_num_list,
#                                                                  voxel_size=voxel_size,
#                                                                  padding_value=padding_value,
#                                                                  pooling_size=pooling_size)
#     return output_features
#
# @ops.RegisterGradient("LaRoiPoolingOp")
# def la_roi_pooling_grad(op, grad, _, __):
#     input_features = op.inputs[1]
#     output_idx = op.outputs[1]
#     output_weight = op.outputs[2]
#     input_features_grad = la_roi_pooling_exe.la_roi_pooling_grad_op(input_features=input_features,
#                                                                     output_idx=output_idx,
#                                                                     output_weight=output_weight,
#                                                                     output_features_grad=grad)
#     return [None, input_features_grad, None, None, None]
#
# # =============================================La Roi Pooling Fast===============================================
#
la_roi_pooling_fast_exe = tf.load_op_library(join(CWD, 'build', 'la_roi_pooling_fast.so'))
def la_roi_pooling_fast(input_coors, input_features, roi_attrs, input_num_list, roi_num_list,
                        dimension, offset, grid_buffer_resolution=0.8,
                        grid_buffer_size=4, voxel_size=5, padding_value=0., pooling_size=5):
    output_features, idx, weight = la_roi_pooling_fast_exe.la_roi_pooling_fast_op(input_coors=input_coors + offset,
                                                                           input_features=input_features,
                                                                           roi_attrs=roi_attrs,
                                                                           input_num_list=input_num_list,
                                                                           roi_num_list=roi_num_list,
                                                                           voxel_size=voxel_size,
                                                                           padding_value=padding_value,
                                                                           pooling_size=pooling_size,
                                                                           dimension=dimension,
                                                                           offset=offset,
                                                                           grid_buffer_resolution=grid_buffer_resolution,
                                                                           grid_buffer_size=grid_buffer_size)
    return output_features, idx, weight
ops.NoGradient("LaRoiPoolingFastOp")

def la_roi_pooling_fast_grad(features, idx, weight, voxels):
    grad = la_roi_pooling_fast_exe.la_roi_pooling_fast_grad_op(input_features=features,
                                                               output_idx=idx,
                                                               output_weight=weight,
                                                               output_features_grad=voxels)
    return grad
#
# # =============================================RoI Filter===============================================
#
# roi_filter_exe = tf.load_op_library(join(CWD, 'build', 'roi_filter.so'))
# def roi_filter(input_roi_attrs, input_roi_conf, input_num_list, conf_thres, max_length, with_negative):
#     output_num_list, output_idx = roi_filter_exe.roi_filter_op(input_roi_conf=input_roi_conf,
#                                                                input_num_list=input_num_list,
#                                                                conf_thres=conf_thres,
#                                                                max_length=max_length,
#                                                                with_negative=with_negative)
#     output_roi_attrs = tf.gather(input_roi_attrs, output_idx, axis=0)
#     return output_roi_attrs, output_num_list, output_idx
# ops.NoGradient("RoiFilterOp")
#
#
# # =============================================Voxel2Col===============================================
#
# voxel_to_col_exe = tf.load_op_library(join(CWD, 'build', 'voxel2col.so'))
# def voxel2col(input_voxels, kernel_size=3):
#     channels = input_voxels.shape[2]
#     output_voxels, _ = voxel_to_col_exe.voxel_to_col_op(input_voxels=input_voxels,
#                                                         kernel_size=kernel_size,
#                                                         channels=channels)
#     return output_voxels
#
# @ops.RegisterGradient("VoxelToColOp")
# def voxel2col_grad(op, grad, _):
#     input_voxels = op.inputs[0]
#     output_idx = op.outputs[1]
#     input_voxels_grad = voxel_to_col_exe.voxel_to_col_grad_op(input_voxels=input_voxels,
#                                                               output_idx=output_idx,
#                                                               output_voxels_grad=grad)
#     return input_voxels_grad
#
#
# # =============================================Roi Logits To Attrs===============================================
#
# roi_logits_to_attrs_exe = tf.load_op_library(join(CWD, 'build', 'roi_logits_to_attrs.so'))
# def roi_logits_to_attrs(base_coors, input_logits, anchor_size):
#     output_attrs = roi_logits_to_attrs_exe.roi_logits_to_attrs_op(base_coors=base_coors,
#                                                                   input_logits=input_logits,
#                                                                   anchor_size=anchor_size)
#     return output_attrs
# ops.NoGradient("RoiLogitsToAttrs")
#
# # =============================================Bbox Logits To Attrs===============================================
#
# bbox_logits_to_attrs_exe = tf.load_op_library(join(CWD, 'build', 'bbox_logits_to_attrs.so'))
# def bbox_logits_to_attrs(input_roi_attrs, input_logits):
#     output_attrs = bbox_logits_to_attrs_exe.bbox_logits_to_attrs_op(input_roi_attrs=input_roi_attrs,
#                                                                     input_logits=input_logits)
#     return output_attrs
# ops.NoGradient("BboxLogitsToAttrs")
#
#
# # =============================================Grid Sampling Thrust===============================================
#
# unique_exe = tf.load_op_library(join(CWD, 'build', 'unique.so'))
# def grid_sampling_thrust(input_coors,
#                         input_num_list,
#                         resolution,
#                         dimension=[70.4, 80.0, 4.0],
#                         offset=[0., 40.0, 3.0]):
#     '''
#     The grid sub-sampling strategy, aims at taking the place of FPS. This operation intents to yield uniformly distributed
# sampling result. This operation is implemented in stack style, which means the number of points of each input instance
#     does not have to be fixed number, and difference instances are differentiated using "input_num_list".
#
#     :param input_coors: 2-D tf.float32 Tensor with shape=[input_npoint, channels].
#     :param input_num_list: 1-D tf.int32 Tensor with shape=[batch_size], indicating how many points within each instance.
#     :param resolution: float32, the down sampleing resolution.
#     :param dimension: 1-D float32 list with shape 3, the maximum in x, y, z orientation of the input coors, this will be used to
#                       create the unique voxel ids for each input points
#     :param offset: 1-D float32 list with shape 3, the offset on each axis, so that the minimum coors in each axis is > 0.
#     :return:
#     output_coors: 2-D tf.float32 Tensor with shape=[output_npoint, channels], the output coordinates of the sub-sampling.
#     output_num_list: 1-D tf.int32 Tensor with shape=[batch_size], same definition as input_num_list.
#     '''
#
#     npoint = tf.shape(input_coors)[0]
#     batch_size = tf.shape(input_num_list)[0]
#     dim_w = tf.cast(tf.floor(dimension[0] / resolution), dtype=tf.int64)
#     dim_l = tf.cast(tf.floor(dimension[1] / resolution), dtype=tf.int64)
#     dim_h = tf.cast(tf.floor(dimension[2] / resolution), dtype=tf.int64)
#     dim_offset = dim_w * dim_l * dim_h
#
#     point_ids = tf.range(npoint) + 1
#     point_ids_array = tf.cast(tf.tile(tf.expand_dims(point_ids, axis=0), [batch_size, 1]), dtype=tf.float32)
#     accu_num_list = tf.cast(tf.cumsum(input_num_list), dtype=tf.float32)
#     masks = tf.cast(tf.greater(point_ids_array / tf.expand_dims(accu_num_list, axis=-1), 1.0), dtype=tf.int64)
#     voxel_offset_masks = tf.reduce_sum(masks, axis=0) * dim_offset
#
#     input_voxel_coors = tf.cast(tf.floor((input_coors + offset) / resolution), dtype=tf.int64)
#     input_voxel_ids = input_voxel_coors[:, 2] * dim_l * dim_w + input_voxel_coors[:, 1] * dim_w + input_voxel_coors[:, 0]
#     input_voxel_ids += voxel_offset_masks
#     input_point_ids = tf.range(npoint, dtype=tf.int32)
#
#     unique_point_ids = unique_exe.unique_op(input_voxel_ids=input_voxel_ids,
#                                             input_point_ids=input_point_ids)
#
#     unique_coors = tf.gather(input_coors, unique_point_ids, axis=0)
#     unique_voxels = tf.gather(input_voxel_ids, unique_point_ids, axis=0)
#
#     voxel_batch_id = tf.cast(tf.floor(unique_voxels / dim_offset), dtype=tf.int32)
#     batch_array = tf.cast(tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, tf.shape(unique_voxels)[0]]), dtype=tf.int32)
#     output_num_list = tf.reduce_sum(tf.cast(tf.equal(voxel_batch_id, batch_array), dtype=tf.int32), axis=-1)
#
#
#     # unique_voxels_array = tf.cast(tf.tile(tf.expand_dims(unique_voxels, 0), [batch_size, 1]), dtype=tf.float32)
#     # bottom_offset_list = tf.cast(tf.range(batch_size), dtype=tf.float32) * tf.to_float(dim_offset)
#     # upper_offset_list = tf.cast(tf.range(batch_size) + 1, dtype=tf.float32) * tf.to_float(dim_offset)
#     # bottom_masks = tf.cast(tf.greater(unique_voxels_array / tf.expand_dims(bottom_offset_list, axis=-1), 1.0), dtype=tf.float32)
#     # up_masks = tf.cast(tf.greater(unique_voxels_array / tf.expand_dims(upper_offset_list, axis=-1), 1.0), dtype=tf.float32)
#     # bottom_count = tf.reduce_sum(bottom_masks, axis=-1)
#     # up_count = tf.reduce_sum(up_masks, axis=-1)
#     # output_num_list = tf.cast(bottom_count - up_count, tf.int32)
#
#     return unique_coors, output_num_list
#
# ops.NoGradient("UniqueOp")
#
# # =============================================Voxel Sampling Binary===============================================
#
# voxel_sampling_binary_exe = tf.load_op_library(join(CWD, 'build', 'voxel_sampling_binary.so'))
# def voxel_sampling_binary(input_coors,
#                          input_features,
#                          input_num_list,
#                          center_coors,
#                          center_num_list,
#                          resolution,
#                          padding=0.,
#                          dimension=[70.4, 80.0, 4.0],
#                          offset=[0., 40.0, 3.0]):
#
#     npoint = tf.shape(input_coors)[0]
#     batch_size = tf.shape(input_num_list)[0]
#     dim_w = tf.cast(tf.floor(dimension[0] / resolution), dtype=tf.int64)
#     dim_l = tf.cast(tf.floor(dimension[1] / resolution), dtype=tf.int64)
#     dim_h = tf.cast(tf.floor(dimension[2] / resolution), dtype=tf.int64)
#     dim_offset = dim_w * dim_l * dim_h
#
#     point_ids = tf.range(npoint) + 1
#     point_ids_array = tf.cast(tf.tile(tf.expand_dims(point_ids, axis=0), [batch_size, 1]), dtype=tf.float32)
#     accu_num_list = tf.cast(tf.cumsum(input_num_list), dtype=tf.float32)
#     masks = tf.cast(tf.greater(point_ids_array / tf.expand_dims(accu_num_list, axis=-1), 1.0), dtype=tf.int64)
#     voxel_offset_masks = tf.reduce_sum(masks, axis=0) * dim_offset
#
#     input_voxel_coors = tf.cast(tf.floor((input_coors + offset) / resolution), dtype=tf.int64)
#     input_voxel_ids = input_voxel_coors[:, 2] * dim_l * dim_w + input_voxel_coors[:, 1] * dim_w + input_voxel_coors[:, 0]
#     input_voxel_ids += voxel_offset_masks
#     sorted_args = tf.argsort(input_voxel_ids)
#     sorted_voxel_ids = tf.gather(input_voxel_ids, sorted_args) - voxel_offset_masks
#     sorted_coors = tf.gather(input_coors, sorted_args, axis=0)
#     sorted_features = tf.gather(input_features, sorted_args, axis=0)
#     # XXX: Need to pay attention to the back-propagation implementation.
#     output_voxels, _ = voxel_sampling_binary_exe.voxel_sampling_binary_op(input_coors=sorted_coors + offset,
#                                                                           input_features=sorted_features,
#                                                                           input_voxel_idx=sorted_voxel_ids,
#                                                                           input_num_list=input_num_list,
#                                                                           center_coors=center_coors + offset,
#                                                                           center_num_list=center_num_list,
#                                                                           dimension=dimension,
#                                                                           resolution=resolution,
#                                                                           padding_value=padding)
#     return output_voxels
#
# @ops.RegisterGradient("VoxelSamplingBinaryOp")
# def voxel_sampling_binary_grad(op, grad, _):
#     input_features = op.inputs[1]
#     output_idx = op.outputs[1]
#     input_features_grad = voxel_sampling_binary_exe.voxel_sampling_binary_grad_op(input_features=input_features,
#                                                                                   output_idx=output_idx,
#                                                                                   output_features_grad=grad)
#     return [None, input_features_grad, None, None, None, None]
#
# # ============================================= NMS ===============================================
#
# iou3d_kernel_gpu_exe = tf.load_op_library(join(CWD, 'build', 'nms.so'))
# def rotated_nms3d(bbox_attrs, bbox_conf, nms_overlap_thresh, nms_conf_thres):
#     '''
#     rotated nms of the output
#     :param boxes: the set of bounding boxes (sorted in decending order based on a score, e.g. confidence)
#                     in [M, 7] := [x, y, z, w, l, h, ry]
#                     where ry = anti-clockwise in Z-up system
#     :param nms_overlap_thresh: The boxes that overlaps with a given box more than the threshold will be remove .
#         threshold range [0,1]
#     Use case:
#     boxes[output_keep_index[:output_num_to_keep]] := gives the list of the valid bounding boxes
#
#     '''
#     valid_idx = tf.where(tf.greater(bbox_conf, nms_conf_thres))[:, 0]
#
#     bbox_attrs = tf.gather(bbox_attrs, valid_idx, axis=0)
#     bbox_conf = tf.gather(bbox_conf, valid_idx, axis=0)
#
#
#     sorted_idx = tf.argsort(bbox_conf, direction='DESCENDING')
#     sorted_bbox_attrs = tf.gather(bbox_attrs, sorted_idx, axis=0)
#     sorted_bbox_conf = tf.gather(bbox_conf, sorted_idx, axis=0)
#
#     bbox_dimensions = sorted_bbox_attrs[:, :3]
#     bbox_coors = sorted_bbox_attrs[:, 3:6]
#     bbox_rotations = sorted_bbox_attrs[:, 6:]
#     bboxes = tf.concat([bbox_coors, bbox_dimensions, bbox_rotations], axis=-1)
#
#     output_keep_index, output_num_to_keep = iou3d_kernel_gpu_exe.rotated_nms3d(
#         input_boxes=bboxes,
#         nms_overlap_thresh=nms_overlap_thresh)
#
#
#
#     return sorted_bbox_attrs, sorted_bbox_conf, output_keep_index, output_num_to_keep
#
#
# ops.NoGradient("RotatedNms3d")
