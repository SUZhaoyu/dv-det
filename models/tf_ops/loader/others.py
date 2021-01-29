import os
from os.path import join

import tensorflow as tf
from tensorflow.python.framework import ops

CWD = os.path.dirname(os.path.abspath(__file__))

# =============================================Voxel2Col===============================================

voxel_to_col_exe = tf.load_op_library(join(CWD, '../build', 'voxel2col.so'))
def voxel2col(input_voxels, kernel_size=3):
    channels = input_voxels.shape[2]
    output_voxels, _ = voxel_to_col_exe.voxel_to_col_op(input_voxels=input_voxels,
                                                        kernel_size=kernel_size,
                                                        channels=channels)
    return output_voxels

@ops.RegisterGradient("VoxelToColOp")
def voxel2col_grad(op, grad, _):
    input_voxels = op.inputs[0]
    output_idx = op.outputs[1]
    input_voxels_grad = voxel_to_col_exe.voxel_to_col_grad_op(input_voxels=input_voxels,
                                                              output_idx=output_idx,
                                                              output_voxels_grad=grad)
    return input_voxels_grad

# ============================================= NMS ===============================================

iou3d_kernel_gpu_exe = tf.load_op_library(join(CWD, '../build', 'nms.so'))
def rotated_nms3d(bbox_attrs, bbox_conf, nms_overlap_thresh, nms_conf_thres):
    '''
    rotated nms of the output
    :param boxes: the set of bounding boxes (sorted in decending order based on a score, e.g. confidence)
                    in [M, 7] := [x, y, z, w, l, h, ry]
                    where ry = anti-clockwise in Z-up system
    :param nms_overlap_thresh: The boxes that overlaps with a given box more than the threshold will be remove .
        threshold range [0,1]
    Use case:
    boxes[output_keep_index[:output_num_to_keep]] := gives the list of the valid bounding boxes

    '''
    valid_idx = tf.where(tf.greater(bbox_conf, nms_conf_thres))[:, 0]

    bbox_attrs = tf.gather(bbox_attrs, valid_idx, axis=0)
    bbox_conf = tf.gather(bbox_conf, valid_idx, axis=0)


    sorted_idx = tf.argsort(bbox_conf, direction='DESCENDING')
    sorted_bbox_attrs = tf.gather(bbox_attrs, sorted_idx, axis=0)
    sorted_bbox_conf = tf.gather(bbox_conf, sorted_idx, axis=0)

    bbox_dimensions = sorted_bbox_attrs[:, :3]
    bbox_coors = sorted_bbox_attrs[:, 3:6]
    bbox_rotations = sorted_bbox_attrs[:, 6:]
    bboxes = tf.concat([bbox_coors, bbox_dimensions, bbox_rotations], axis=-1)

    output_keep_index, output_num_to_keep = iou3d_kernel_gpu_exe.rotated_nms3d(
        input_boxes=bboxes,
        nms_overlap_thresh=nms_overlap_thresh)



    return sorted_bbox_attrs, sorted_bbox_conf, output_keep_index, output_num_to_keep


ops.NoGradient("RotatedNms3d")
