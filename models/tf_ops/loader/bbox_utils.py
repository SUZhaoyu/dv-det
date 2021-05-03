import os
from os.path import join
import numpy as np
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


# =============================================Get Bev Ground Truth===============================================

get_bev_gt_bbox_exe = tf.load_op_library(join(CWD, '../build', 'get_bev_gt_bbox.so'))
def get_bev_gt_bbox(input_coors, label_bbox, input_num_list, anchor_param_list, expand_ratio=0.15, diff_thres=4, cls_thres=1):
    bbox_attrs, bbox_conf, label_idx = get_bev_gt_bbox_exe.get_bev_gt_bbox_op(input_coors=input_coors,
                                                                              label_bbox=label_bbox,
                                                                              input_num_list=input_num_list,
                                                                              anchor_param_list=anchor_param_list,
                                                                              expand_ratio=expand_ratio,
                                                                              diff_thres=diff_thres,
                                                                              cls_thres=cls_thres)
    return bbox_attrs, bbox_conf, label_idx
ops.NoGradient("GetBevGtBboxOp")

# =============================================Roi Logits To Attrs===============================================

roi_logits_to_attrs_exe = tf.load_op_library(join(CWD, '../build', 'roi_logits_to_attrs.so'))
def roi_logits_to_attrs(base_coors, input_logits, anchor_size):
    output_attrs = roi_logits_to_attrs_exe.roi_logits_to_attrs_op(base_coors=base_coors,
                                                                  input_logits=input_logits,
                                                                  anchor_size=anchor_size)
    return output_attrs
ops.NoGradient("RoiLogitsToAttrsOp")

# =============================================Bbox Logits To Attrs===============================================

bbox_logits_to_attrs_exe = tf.load_op_library(join(CWD, '../build', 'bbox_logits_to_attrs.so'))
def bbox_logits_to_attrs(input_roi_attrs, input_logits):
    output_attrs = bbox_logits_to_attrs_exe.bbox_logits_to_attrs_op(input_roi_attrs=input_roi_attrs,
                                                                    input_logits=input_logits)
    return output_attrs
ops.NoGradient("BboxLogitsToAttrsOp")




def get_anchor_attrs(anchor_coors, anchor_param_list):  # [n, 2], [k, f]
    anchor_param_list = tf.expand_dims(anchor_param_list, axis=0)  # [1, k, f]
    anchor_param_list = tf.tile(anchor_param_list, [tf.shape(anchor_coors)[0], 1, 1])  # [n, k, f]
    output_anchor_attrs = []
    for k in range(anchor_param_list.shape[1]):
        anchor_param = anchor_param_list[:, k, :] # [n, f] (w, l, h, z, r)
        anchor_attrs = tf.stack([anchor_param[:, 0],
                                 anchor_param[:, 1],
                                 anchor_param[:, 2],
                                 anchor_coors[:, 0],
                                 anchor_coors[:, 1],
                                 anchor_param[:, 3],
                                 anchor_param[:, 4]], axis=-1) # [n, f]
        output_anchor_attrs.append(anchor_attrs)

    return tf.stack(output_anchor_attrs, axis=1) # [n, k, f]


def logits_to_attrs(anchor_coors, input_logits, anchor_param_list): # [n, k, f]
    output_attrs = []
    # anchor_param_list = tf.expand_dims(anchor_param_list, axis=0) # [k, f]
    for k in range(anchor_param_list.shape[0]):
        anchor_param = anchor_param_list[k, :] # [f]
        anchor_diag = tf.sqrt(tf.pow(anchor_param[0], 2.) + tf.pow(anchor_param[1], 2.))
        w = tf.clip_by_value(tf.exp(input_logits[:, k, 0]) * anchor_param[0], 0., 1e7)
        l = tf.clip_by_value(tf.exp(input_logits[:, k, 1]) * anchor_param[1], 0., 1e7)
        h = tf.clip_by_value(tf.exp(input_logits[:, k, 2]) * anchor_param[2], 0., 1e7)
        x = tf.clip_by_value(input_logits[:, k, 3] * anchor_diag + anchor_coors[:, 0], -1e7, 1e7)
        y = tf.clip_by_value(input_logits[:, k, 4] * anchor_diag + anchor_coors[:, 1], -1e7, 1e7)
        z = tf.clip_by_value(input_logits[:, k, 5] * anchor_param[2] + anchor_param[3], -1e7, 1e7)
        r = tf.clip_by_value((input_logits[:, k, 6] + anchor_param[4]) * np.pi, -1e7, 1e7)
        output_attrs.append(tf.stack([w, l, h, x, y, z, r], axis=-1)) # [n, f]
    return tf.stack(output_attrs, axis=1) # [n, k, f]

