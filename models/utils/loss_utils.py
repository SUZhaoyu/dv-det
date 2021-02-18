import numpy as np
import tensorflow as tf


def focal_loss(label, pred, alpha=0.25, gamma=2):
    part_a = -alpha * (1 - pred) ** gamma * tf.log(pred) * label
    part_b = -(1 - alpha) * pred ** gamma * tf.log(1 - pred) * (1 - label)
    return part_a + part_b


def smooth_l1_loss(predictions, labels, delta=1.0):
    residual = tf.abs(tf.sin(predictions - labels))
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


