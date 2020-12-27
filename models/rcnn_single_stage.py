import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf

from models.tf_ops.custom_ops import get_roi_bbox
from models.utils.iou_utils import cal_3d_iou, roi_logits_to_attrs_tf
from models.utils.model_layers import point_conv, fully_connected

anchor_size = tf.constant([1.6, 3.9, 1.5])
eps = tf.constant(1e-6)


def inputs_placeholder(input_channels=1,
                       bbox_padding=64):
    input_coors_p = tf.placeholder(tf.float32, shape=[None, None], name='input_coors_p')
    input_features_p = tf.placeholder(tf.float32, shape=[None, input_channels], name='input_features_p')
    input_num_list_p = tf.placeholder(tf.int32, shape=[None], name='input_num_list_p')
    input_bbox_p = tf.placeholder(dtype=tf.float32, shape=[None, bbox_padding, 9], name='input_bbox_p')
    return input_coors_p, input_features_p, input_num_list_p, input_bbox_p


def model(input_coors,
          input_features,
          input_num_list,
          is_training,
          config,
          bn):
    model_params = {'xavier': config.xavier,
                    'stddev': config.stddev,
                    'activation': config.activation,
                    'dimension': config.dimension,
                    'offset': config.offset}
    base_params = config.base_params

    with tf.variable_scope("rcnn"):
        coors, features, num_list = input_coors, input_features, input_num_list
        for layer_name in sorted(base_params.keys()):
            coors, features, num_list = point_conv(input_coors=coors,
                                                   input_features=features,
                                                   input_num_list=num_list,
                                                   layer_params=base_params[layer_name],
                                                   scope=layer_name,
                                                   is_training=is_training,
                                                   model_params=model_params,
                                                   bn_decay=bn)

        base_coors, roi_features, roi_num_list = point_conv(input_coors=coors,
                                                            input_features=features,
                                                            input_num_list=num_list,
                                                            layer_params=config.rpn_params,
                                                            scope="rpn_conv",
                                                            is_training=is_training,
                                                            model_params=model_params,
                                                            bn_decay=bn)

        roi_logits = fully_connected(input_points=roi_features,
                                     num_output_channels=config.output_attr,
                                     drop_rate=0.,
                                     model_params=model_params,
                                     scope='rpn_fc',
                                     is_training=is_training,
                                     last_layer=True)

        roi_attrs = roi_logits_to_attrs_tf(input_logits=roi_logits,
                                           base_coors=base_coors,
                                           anchor_size=anchor_size,
                                           norm_angle=config.norm_angle)
        roi_conf_logits = roi_logits[:, 7]

        return base_coors, roi_attrs, roi_conf_logits, roi_num_list


def focal_loss(label, pred, alpha=0.25, gamma=2):
    part_a = -alpha * (1 - pred) ** gamma * tf.log(pred) * label
    part_b = -(1 - alpha) * pred ** gamma * tf.log(1 - pred) * (1 - label)
    return part_a + part_b


def smooth_l1_loss(predictions, labels, delta=1.0, use_sin=False):
    if use_sin:
        residual = tf.abs(tf.sin(predictions - labels))
    else:
        residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def get_masked_average(input, mask):
    return tf.math.divide_no_nan(tf.reduce_sum(input * mask), tf.reduce_sum(mask))


def get_loss(base_coors, pred_roi_attrs, pred_roi_conf_logits, num_list, bbox_labels, wd=0.):
    pred_roi_conf = tf.nn.sigmoid(pred_roi_conf_logits)
    pred_roi_conf = tf.clip_by_value(pred_roi_conf, eps, 1 - eps)

    gt_roi_attrs, gt_roi_conf, gt_roi_diff = get_roi_bbox(input_coors=base_coors,
                                                          bboxes=bbox_labels,
                                                          input_num_list=num_list,
                                                          anchor_size=anchor_size,
                                                          expand_ratio=0.1,
                                                          diff_thres=2)

    roi_ious = cal_3d_iou(gt_attrs=gt_roi_attrs, pred_attrs=pred_roi_attrs, clip=False)

    diff_masks_easy = tf.cast(tf.equal(gt_roi_diff, 0), dtype=tf.float32)
    diff_masks_moderate = tf.cast(tf.equal(gt_roi_diff, 1), dtype=tf.float32)
    diff_masks_hard = tf.cast(tf.equal(gt_roi_diff, 2), dtype=tf.float32)

    roi_conf_masks = tf.cast(tf.greater(gt_roi_conf, -1), dtype=tf.float32)  # [-1, 0, 1] -> [0, 1, 1]
    roi_iou_masks = tf.cast(tf.equal(gt_roi_conf, 1), dtype=tf.float32)  # [-1, 0, 1] -> [0, 0, 1]
    roi_iou_eval_masks = tf.cast(tf.greater(pred_roi_conf, 0.5), dtype=tf.float32)
    roi_conf_target = tf.cast(gt_roi_conf, dtype=tf.float32) * roi_conf_masks  # [-1, 0, 1] * [0, 1, 1] -> [0, 0, 1]

    tf.summary.histogram('roi_conf_target', roi_conf_target)

    roi_conf_loss = get_masked_average(focal_loss(label=roi_conf_target, pred=pred_roi_conf, alpha=0.25),
                                       roi_conf_masks)
    roi_conf_acc = tf.cast(tf.equal(roi_conf_target, roi_iou_eval_masks), dtype=tf.float32)
    tf.summary.scalar('roi_conf_loss', roi_conf_loss)

    tf.summary.scalar('roi_conf_acc_easy', get_masked_average(roi_conf_acc, diff_masks_easy))
    tf.summary.scalar('roi_conf_acc_moderate', get_masked_average(roi_conf_acc, diff_masks_moderate))
    tf.summary.scalar('roi_conf_acc_hard', get_masked_average(roi_conf_acc, diff_masks_hard))

    roi_iou_loss = get_masked_average(1 - roi_ious, roi_iou_masks)

    average_roi_iou_easy = get_masked_average(roi_ious, diff_masks_easy)
    average_roi_iou_moderate = get_masked_average(roi_ious, diff_masks_moderate)
    average_roi_iou_hard = get_masked_average(roi_ious, diff_masks_hard)
    tf.summary.scalar('roi_iou_easy', average_roi_iou_easy)
    tf.summary.scalar('roi_iou_moderate', average_roi_iou_moderate)
    tf.summary.scalar('roi_iou_hard', average_roi_iou_hard)

    average_roi_iou = tf.reduce_mean([average_roi_iou_easy, average_roi_iou_moderate, average_roi_iou_hard])
    tf.summary.scalar('roi_iou_loss', roi_iou_loss)
    tf.summary.scalar('roi_iou', average_roi_iou)

    roi_angle_loss = get_masked_average(smooth_l1_loss(pred_roi_attrs[:, 6], gt_roi_attrs[:, 6], use_sin=True),
                                        roi_iou_masks)
    average_roi_angle = get_masked_average(tf.abs(tf.floormod(pred_roi_attrs[:, 6] - gt_roi_attrs[:, 6], np.pi)),
                                           roi_iou_eval_masks)
    tf.summary.scalar('average_roi_angle_delta', average_roi_angle)
    tf.summary.scalar('roi_angle_loss', roi_angle_loss)

    roi_l2_loss = wd * tf.add_n(tf.get_collection("l2"))
    tf.summary.scalar('roi_l2', roi_l2_loss)

    roi_loss = roi_conf_loss + 1.5 * roi_iou_loss + roi_l2_loss + roi_angle_loss

    roi_iou_collection = hvd.allreduce(average_roi_iou)
    roi_loss_collection = hvd.allreduce(roi_loss)

    tf.summary.scalar('roi_loss_average', roi_loss_collection)
    tf.summary.scalar('roi_iou_average', roi_iou_collection)

    return roi_loss, roi_iou_collection
