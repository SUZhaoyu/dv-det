import numpy as np
import tensorflow as tf

from models.tf_ops.custom_ops import roi_pooling, voxel_sampling, grid_sampling
from models.utils.ops_wrapper import dense_conv_wrapper, kernel_conv_wrapper, fully_connected_wrapper

def point_conv(input_coors, input_features, input_num_list, c_out, resolution, scope):
    coors, num_list = grid_sampling(input_coors=input_coors,
                                    input_num_list=input_num_list,
                                    resolution=resolution)

    voxels = voxel_sampling(input_coors=input_coors,
                            input_features=input_features,
                            input_num_list=input_num_list,
                            center_coors=coors,
                            center_num_list=num_list,
                            resolution=resolution,
                            padding=-1)

    features = kernel_conv_wrapper(inputs=voxels,
                                   num_output_channels=c_out,
                                   scope=scope,
                                   bn_decay=1.)

    return coors, features, num_list


def conv_3d(input_voxels, c_out, scope):
    output_features = dense_conv_wrapper(inputs=input_voxels,
                                         num_output_channels=c_out,
                                         kernel_size=3,
                                         scope=scope,
                                         bn_decay=1.)
    return output_features


def fully_connected(input_points, c_out, scope):
    output_points = fully_connected_wrapper(inputs=input_points,
                                            num_output_channels=c_out,
                                            scope=scope,
                                            bn_decay=1.)
    return output_points

def get_roi_attrs_from_logits(input_logits, base_coors, anchor_size):
    anchor_diag = tf.sqrt(tf.pow(anchor_size[0], 2.) + tf.pow(anchor_size[1], 2.))
    w = tf.clip_by_value(tf.exp(input_logits[:, 0]) * anchor_size[0], 0., 1e5)
    l = tf.clip_by_value(tf.exp(input_logits[:, 1]) * anchor_size[1], 0., 1e5)
    h = tf.clip_by_value(tf.exp(input_logits[:, 2]) * anchor_size[2], 0., 1e5)
    x = tf.clip_by_value(input_logits[:, 3] * anchor_diag + base_coors[:, 0], -1e5, 1e5)
    y = tf.clip_by_value(input_logits[:, 4] * anchor_diag + base_coors[:, 1], -1e5, 1e5)
    z = tf.clip_by_value(input_logits[:, 5] * anchor_size[2] + base_coors[:, 2], -1e5, 1e5)
    r = input_logits[:, 6]
    return tf.stack([w, l, h, x, y, z, r], axis=-1)

def get_bbox_attrs_from_logits(input_logits, roi_attrs):
    roi_diag = tf.sqrt(tf.pow(roi_attrs[:, 0], 2.) + tf.pow(roi_attrs[:, 1], 2.))
    w = tf.clip_by_value(tf.exp(input_logits[:, 0]) * roi_attrs[:, 0], 0., 1e5)
    l = tf.clip_by_value(tf.exp(input_logits[:, 1]) * roi_attrs[:, 1], 0., 1e5)
    h = tf.clip_by_value(tf.exp(input_logits[:, 2]) * roi_attrs[:, 2], 0., 1e5)
    x = tf.clip_by_value(input_logits[:, 3] * roi_diag + roi_attrs[:, 3], -1e5, 1e5)
    y = tf.clip_by_value(input_logits[:, 4] * roi_diag + roi_attrs[:, 4], -1e5, 1e5)
    z = tf.clip_by_value(input_logits[:, 5] * roi_attrs[:, 2] + roi_attrs[:, 5], -1e5, 1e5)
    r = input_logits[:, 6] + roi_attrs[:, 6]
    return tf.stack([w, l, h, x, y, z, r], axis=-1)