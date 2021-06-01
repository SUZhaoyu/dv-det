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
    label_dimension = tf.clip_by_value(label_dimension, 1e-6, 1e6)
    target_logits = tf.log(label_dimension / anchor_size)
    return target_logits

def get_target_bin(point_coors, target_coors, scope, delta, masks, name=None):
    if name is not None:
        tf.summary.histogram(name+"_offset", (target_coors - point_coors))
    num_bin = int(scope / delta) * 2
    target_bin = tf.cast(tf.floor((target_coors - point_coors + scope) / delta), dtype=tf.int32)
    if name is not None:
        tf.summary.histogram(name, target_bin * tf.expand_dims(tf.cast(masks, dtype=tf.int32), axis=-1))
    target_bin = tf.clip_by_value(target_bin, 0, num_bin-1)
    return target_bin

def get_residual(point_coors, target_coors, target_bin, scope, delta, masks, name=None):
    residual = target_coors - point_coors + scope - (tf.cast(target_bin, dtype=tf.float32) * delta) - 0.5 * delta
    residual /= delta
    if name is not None:
        tf.summary.histogram(name, residual * tf.expand_dims(tf.cast(masks, dtype=tf.float32), axis=-1))
    return residual

def get_bbox_loss(point_coors,
                  pred_logits,
                  label_attrs,
                  foreground_masks,
                  anchor_size,
                  scope=3.,
                  delta_bin_xy=0.5,
                  delta_bin_angle=np.pi / 6.): # [w, l, h, x, y, z, r]
    '''
    pred_logits=[(w, l, h), (res_x, res_y, res_z, res_r), (cls_x, cls_y, cls_r) x 12]
                  0, 1, 2 ,    3  ,   4  ,   5   ,  6,     7-18,  19-30, 31-42
    '''
    target_bin_xy = get_target_bin(point_coors[:, :2], label_attrs[:, 3:5], scope, delta_bin_xy, foreground_masks, name="cls_bin_xy") # [n, 2]
    # target_bin_r = get_target_bin(tf.zeros_like(point_coors[:, :1]), label_attrs[:, 6:], np.pi, delta_bin_angle, name="cls_bin_r")
    target_residual_xy = get_residual(point_coors[:, :2], label_attrs[:, 3:5], target_bin_xy, scope, delta_bin_xy, foreground_masks, name="res_bin_xy")
    target_residual_z = label_attrs[:, 5:6] - point_coors[:, 2:3]
    # target_residual_r = get_residual(tf.zeros_like(label_attrs[:, 6:7]), label_attrs[:, 6:7], target_bin_r, np.pi, delta_bin_angle, name="res_bin_r")
    target_residual_dimension = get_target_dimension_logits(label_attrs[:, :3], anchor_size)

    # cls_target = tf.concat([target_bin_xy, target_bin_r], axis=-1)
    cls_target = target_bin_xy
    res_target = tf.concat([target_residual_dimension, target_residual_xy, target_residual_z], axis=-1)

    pred_cls_logits = tf.stack([pred_logits[:, 7:19], pred_logits[:, 19:31]], axis=1) # [n, 3, 12]
    cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=cls_target, logits=pred_cls_logits) # [n, 3]
    pred_res_logits = pred_logits[:, :6]

    cls_acc = get_masked_average(tf.reduce_mean(tf.cast(tf.equal(cls_target, tf.cast(tf.argmax(tf.nn.softmax(pred_cls_logits), axis=-1), dtype=tf.int32)), dtype=tf.float32), axis=-1), foreground_masks)
    tf.summary.scalar("logits_cls_acc", cls_acc)
    tf.summary.scalar("dimension_avg_diff", get_masked_average(tf.reduce_mean(tf.abs(target_residual_dimension - pred_res_logits[:, :3]), axis=-1), foreground_masks))
    tf.summary.scalar("xy_res_avg_diff", get_masked_average(tf.reduce_mean(tf.abs(target_residual_xy - pred_res_logits[:, 3:5]), axis=-1), foreground_masks))
    tf.summary.scalar("z_res_avg_diff", get_masked_average(tf.reduce_mean(tf.abs(target_residual_z - pred_res_logits[:, 5:6]), axis=-1), foreground_masks))

    pred_angle = pred_logits[:, 6:7] * np.pi
    angle_loss = smooth_l1_loss(pred_angle, label_attrs[:, 6:7], with_sin=True)
    reg_loss = smooth_l1_loss(pred_res_logits, res_target, with_sin=False) # [n, 6]


    bbox_loss = tf.concat([cls_loss, reg_loss, angle_loss], axis=-1) # [n, 9]
    bbox_loss_sum = tf.reduce_sum(bbox_loss, axis=-1) # [n]

    return get_masked_average(bbox_loss_sum, foreground_masks)


def get_bbox_from_logits(point_coors,
                         pred_logits,
                         anchor_size,
                         scope=3.,
                         delta_bin_xy=0.5,
                         delta_bin_angle=np.pi / 6.):
    pred_dim = tf.exp(pred_logits[:, :3]) * anchor_size
    res_xy = pred_logits[:, 3:5] * delta_bin_xy
    res_r = pred_logits[:, 6:7] * np.pi
    pred_offset_z = pred_logits[:, 5:6]
    cls_logits = tf.stack([pred_logits[:, 7:19], pred_logits[:, 19:31]], axis=1)
    bin_cls = tf.cast(tf.argmax(cls_logits, axis=-1), dtype=tf.float32)  # [n, 3]
    pred_offset_xy = -scope + delta_bin_xy * bin_cls[:, :2] + res_xy + 0.5 * delta_bin_xy
    # pred_angle = -np.pi + delta_bin_angle * bin_cls[:, 2:] + res_r
    pred_angle = res_r

    pred_coors = tf.concat([pred_offset_xy, pred_offset_z], axis=-1) + point_coors

    return tf.concat([pred_dim, pred_coors, pred_angle], axis=-1)


def get_bbox_target_params(point_coors,
                          label_attrs,
                          anchor_size,
                          scope=3.,
                          delta_bin_xy=0.5,
                          delta_bin_angle=np.pi / 6.): # [w, l, h, x, y, z, r]
    '''
    pred_logits=[(w, l, h), (res_x, res_y, res_z), (cls_x, cls_y, cls_r) x 12]
                  0, 1, 2 ,    3  ,   4  ,   5   ,   6-17, 18-29, 30-41
    '''
    target_bin_xy = get_target_bin(point_coors[:, :2], label_attrs[:, 3:5], scope, delta_bin_xy, name="cls_bin_xy") # [n, 2]
    target_bin_r = get_target_bin(tf.zeros_like(point_coors[:, :1]), label_attrs[:, 6:], np.pi, delta_bin_angle, name="cls_bin_r")
    target_residual_xy = get_residual(point_coors[:, :2], label_attrs[:, 3:5], target_bin_xy, scope, delta_bin_xy, name="res_bin_xy")
    target_residual_z = label_attrs[:, 5:6] - point_coors[:, 2:3]
    target_residual_r = get_residual(tf.zeros_like(label_attrs[:, 6:7]), label_attrs[:, 6:7], target_bin_r, np.pi, delta_bin_angle, name="res_bin_r")
    target_residual_dimension = get_target_dimension_logits(label_attrs[:, :3], anchor_size)

    cls_target = tf.concat([target_bin_xy, target_bin_r], axis=-1)
    res_target = tf.concat([target_residual_dimension, target_residual_xy, target_residual_z, target_residual_r], axis=-1)

    return cls_target, res_target
