import os
from os.path import join

import tensorflow as tf
from tensorflow.python.framework import ops

CWD = os.path.dirname(os.path.abspath(__file__))

grid_sampling_exe = tf.load_op_library(join(CWD, 'build', 'grid_sampling.so'))
def grid_sampling(input_coors,
                  input_num_list,
                  resolution,
                  dimension=[70.4, 80.0, 4.0],
                  offset=[0., 40.0, 3.0]):
    output_idx, output_num_list = grid_sampling_exe.grid_sampling_op(input_coors=input_coors + offset,
                                                                     input_num_list=input_num_list,
                                                                     dimension=dimension,
                                                                     resolution=resolution)
    output_coors = tf.gather(input_coors, output_idx, axis=0)
    return output_coors, output_num_list


ops.NoGradient("GridSamplingOp")



# =============================================Voxel Sampling===============================================
voxel_sampling_exe = tf.load_op_library(join(CWD, 'build', 'voxel_sampling.so'))


def voxel_sampling(input_coors,
                   input_features,
                   input_num_list,
                   center_coors,
                   center_num_list,
                   resolution,
                   padding=0.,
                   dimension=[70.4, 80.0, 4.0],
                   offset=[0., 40.0, 3.0]):
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




# =============================================Get RoI Ground Truth===============================================

get_roi_bbox_exe = tf.load_op_library(join(CWD, 'build', 'get_roi_bbox.so'))


def get_roi_bbox(input_coors, bboxes, input_num_list, anchor_size, expand_ratio=0.15, diff_thres=3):
    '''
    Get point-wise RoI ground truth.
    :param input_coors: 2-D Tensor with shape [npoint, 3]
    :param bboxes: 3-D Tensor with shape [batch, nbbox, bbox_attr]
    :param input_num_list: 1-D Tensor with shape [batch]
    :param anchor_size: 1-D list or tensor with shape [4] (w, l, h, z)
    :param channels: int, how many attributes in the final roi output.
    :param expand_ratio: default=0.1
    :param diff_thres: default=3, only the points with difficulty <= diff_thres will be linked to the final loss
    :return: output_attrs: 2-D Tensor with shape [npoint, 10]
                           [confidence, w, l, h, offset_x, offset_y, offset_z, angle, *face_direction(binary), *class]
    '''
    roi_attrs, roi_conf, roi_diff = get_roi_bbox_exe.get_roi_bbox_op(input_coors=input_coors,
                                                                      gt_bbox=bboxes,
                                                                      input_num_list=input_num_list,
                                                                      anchor_size=anchor_size,
                                                                      expand_ratio=expand_ratio,
                                                                      diff_thres=diff_thres)
    return roi_attrs, roi_conf, roi_diff


ops.NoGradient("GetRoiBboxOp")

# =============================================Get Bbox Ground Truth===============================================

get_bbox_exe = tf.load_op_library(join(CWD, 'build', 'get_bbox.so'))
def get_bbox(roi_bbox, bboxes, input_num_list, expand_ratio=0.15, diff_thres=3):
    bbox_attrs, bbox_conf = get_bbox_exe.get_bbox_op(roi_bbox=roi_bbox,
                                                      gt_bbox=bboxes,
                                                      input_num_list=input_num_list,
                                                      expand_ratio=expand_ratio,
                                                      diff_thres=diff_thres)
    return bbox_attrs, bbox_conf
ops.NoGradient("GetBboxOp")

# =============================================Roi Pooling===============================================

roi_pooling_exe = tf.load_op_library(join(CWD, 'build', 'roi_pooling.so'))
def roi_pooling(input_coors, input_features, roi_attrs, input_num_list, rois_num_list,
                voxel_size=5, padding_value=0., pooling_size=5):
    output_features, output_idx = roi_pooling_exe.roi_pooling_op(input_coors=input_coors,
                                                                 input_features=input_features,
                                                                 roi_attrs=roi_attrs,
                                                                 input_num_list=input_num_list,
                                                                 rois_num_list=rois_num_list,
                                                                 voxel_size=voxel_size,
                                                                 padding_value=padding_value,
                                                                 pooling_size=pooling_size)
    return output_features

@ops.RegisterGradient("RoiPoolingOp")
def roi_pooling_grad(op, grad, _):
    input_features = op.inputs[1]
    output_idx = op.outputs[1]
    input_features_grad = roi_pooling_exe.roi_pooling_grad_op(output_idx=output_idx,
                                                              input_features=input_features,
                                                              output_features_grad=grad)
    return [None, input_features_grad, None, None, None]


# =============================================RoI Filter===============================================

roi_filter_exe = tf.load_op_library(join(CWD, 'build', 'roi_filter.so'))
def roi_filter(input_roi_attrs, input_roi_conf, input_num_list, conf_thres):
    output_num_list, output_idx = roi_filter_exe.roi_filter_op(input_roi_conf=input_roi_conf,
                                                               input_num_list=input_num_list,
                                                               conf_thres=conf_thres)
    output_roi_attrs = tf.gather(input_roi_attrs, output_idx, axis=0)
    return output_roi_attrs, output_num_list
ops.NoGradient("RoiFilterOp")


# =============================================Dense Conv===============================================

voxel_to_col_exe = tf.load_op_library(join(CWD, 'build', 'voxel2col.so'))
def voxel2col(input_voxels, kernel_size=3):
    output_voxels, _ = voxel_to_col_exe.voxel_to_col_op(input_voxels=input_voxels,
                                                        kernel_size=kernel_size)
    return output_voxels
ops.NoGradient("DenseConvOp")


# =============================================Roi Logits To Attrs===============================================

roi_logits_to_attrs_exe = tf.load_op_library(join(CWD, 'build', 'roi_logits_to_attrs.so'))
def roi_logits_to_attrs(base_coors, input_logits, anchor_size):
    output_attrs = roi_logits_to_attrs_exe.roi_logits_to_attrs_op(base_coors=base_coors,
                                                                  input_logits=input_logits,
                                                                  anchor_size=anchor_size)
    return output_attrs
ops.NoGradient("RoiLogitsToAttrs")

# =============================================Bbox Logits To Attrs===============================================

bbox_logits_to_attrs_exe = tf.load_op_library(join(CWD, 'build', 'bbox_logits_to_attrs.so'))
def bbox_logits_to_attrs(input_roi_attrs, input_logits):
    output_attrs = bbox_logits_to_attrs_exe.bbox_logits_to_attrs_op(input_roi_attrs=input_roi_attrs,
                                                                    input_logits=input_logits)
    return output_attrs
ops.NoGradient("BboxLogitsToAttrs")