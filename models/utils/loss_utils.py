import numpy as np
import tensorflow as tf

def focal_loss(label, pred, alpha=0.25, gamma=2):
    part_a = -alpha * (1 - pred) ** gamma * tf.log(pred) * label
    part_b = -(1 - alpha) * pred ** gamma * tf.log(1 - pred) * (1 - label)
    return part_a + part_b


def smooth_l1_loss(predictions, labels, delta=1.0, with_sin=True):
    if with_sin:
        residual = tf.abs(tf.sin(predictions - labels))
    else:
        residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def get_masked_average(input, mask):
    return tf.math.divide_no_nan(tf.reduce_sum(input * mask), tf.reduce_sum(mask))

def get_90_rotated_attrs(input_attrs):
    rotated_attrs = tf.stack([input_attrs[:, 1],
                              input_attrs[:, 0],
                              input_attrs[:, 2],
                              input_attrs[:, 3],
                              input_attrs[:, 4],
                              input_attrs[:, 5],
                              input_attrs[:, 6] + np.pi / 2], axis=-1)
    return rotated_attrs

def get_dir_cls(label, pred):
    remainder = tf.math.floormod(tf.abs(label - pred), 2 * np.pi)
    cls = tf.cast(tf.less(tf.cos(remainder), 0.), dtype=tf.float32)
    return cls


def get_target_dimension_logits(label_dimension, anchor_size):
    label_dimension = tf.clip_by_value(1e-6, 1e6, label_dimension)
    target_logits = tf.log(label_dimension / anchor_size)
    return target_logits

def get_target_bin(point_coors, target_coors, scope, num_bin, name=None):
    delta = scope / num_bin
    target_bin = tf.cast(tf.floor((target_coors - point_coors + scope) / delta), dtype=tf.int64)
    target_bin = tf.clip_by_value(0, num_bin-1, target_bin)
    if name is not None:
        tf.summary.histogram(name, target_bin)
    return target_bin

def get_residual(point_coors, target_coors, target_bin, scope, num_bin, name=None):
    delta = scope / num_bin
    residual = target_coors - point_coors + scope - (tf.cast(target_bin, dtype=tf.float32) * delta + 0.5 * delta)
    residual /= delta
    if name is not None:
        tf.summary.histogram(name, residual)
    return residual

def get_bbox_loss(point_coors,
                  pred_logits,
                  label_attrs,
                  foreground_masks,
                  scope,
                  num_bin_offset,
                  num_bin_angle,
                  anchor_size): # [w, l, h, x, y, z, r]
    dimension_logits = get_target_dimension_logits(label_attrs[:, :3], anchor_size) # [n, 3]
    target_bin_xy = get_target_bin(point_coors[:, :2], label_attrs[:, 3:5], scope, num_bin_offset, name="target_bin_xy") # [n, 2]
    target_bin_r = get_target_bin(tf.zeros_like(point_coors[:, :1]), label_attrs[:, 6:], np.pi, num_bin_angle, name="target_bin_r")
