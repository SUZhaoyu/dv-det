import os
from os.path import join

import tensorflow as tf
from tensorflow.python.framework import ops

CWD = os.path.dirname(os.path.abspath(__file__))

# =============================================Get RoI Ground Truth===============================================

get_roi_bbox_exe = tf.load_op_library(join(CWD, '../build', 'get_roi_bbox.so'))


def get_roi_bbox(input_coors, bboxes, input_num_list, anchor_size, expand_ratio=0.15, diff_thres=4, cls_thres=1):
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
                                                                     diff_thres=diff_thres,
                                                                     cls_thres=cls_thres)
    return roi_attrs, roi_conf, roi_diff


ops.NoGradient("GetRoiBboxOp")

# =============================================Get Bbox Ground Truth===============================================

get_bbox_exe = tf.load_op_library(join(CWD, '../build', 'get_bbox.so'))
def get_bbox(roi_attrs, bboxes, input_num_list, expand_ratio=0.15, diff_thres=4, cls_thres=1):
    bbox_attrs, bbox_conf, bbox_diff = get_bbox_exe.get_bbox_op(roi_attrs=roi_attrs,
                                                                gt_bbox=bboxes,
                                                                input_num_list=input_num_list,
                                                                expand_ratio=expand_ratio,
                                                                diff_thres=diff_thres,
                                                                cls_thres=cls_thres)
    return bbox_attrs, bbox_conf, bbox_diff
ops.NoGradient("GetBboxOp")

# =============================================Roi Logits To Attrs===============================================

roi_logits_to_attrs_exe = tf.load_op_library(join(CWD, '../build', 'roi_logits_to_attrs.so'))
def roi_logits_to_attrs(base_coors, input_logits, anchor_size):
    output_attrs = roi_logits_to_attrs_exe.roi_logits_to_attrs_op(base_coors=base_coors,
                                                                  input_logits=input_logits,
                                                                  anchor_size=anchor_size)
    return output_attrs
ops.NoGradient("RoiLogitsToAttrs")

# =============================================Bbox Logits To Attrs===============================================

bbox_logits_to_attrs_exe = tf.load_op_library(join(CWD, '../build', 'bbox_logits_to_attrs.so'))
def bbox_logits_to_attrs(input_roi_attrs, input_logits):
    output_attrs = bbox_logits_to_attrs_exe.bbox_logits_to_attrs_op(input_roi_attrs=input_roi_attrs,
                                                                    input_logits=input_logits)
    return output_attrs
ops.NoGradient("BboxLogitsToAttrs")