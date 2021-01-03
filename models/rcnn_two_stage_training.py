import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf

from models.tf_ops.custom_ops import roi_pooling, get_roi_bbox, roi_filter, get_bbox
from models.utils.iou_utils import cal_3d_iou
from models.utils.model_layers import point_conv, fully_connected, conv_3d
from models.utils.ops_wrapper import get_roi_attrs, get_bbox_attrs
from models.utils.loss_utils import get_masked_average, smooth_l1_loss, focal_loss
import train.configs.rcnn_config as config

anchor_size = [1.6, 3.9, 1.5]
eps = tf.constant(1e-6)

model_params = {'xavier': config.xavier,
                'stddev': config.stddev,
                'activation': config.activation,
                'dimension': config.dimension,
                'offset': config.offset}


def inputs_placeholder(input_channels=1,
                       bbox_padding=64):
    input_coors_p = tf.placeholder(tf.float32, shape=[None, None], name='input_coors_p')
    input_features_p = tf.placeholder(tf.float32, shape=[None, input_channels], name='input_features_p')
    input_num_list_p = tf.placeholder(tf.int32, shape=[None], name='input_num_list_p')
    input_bbox_p = tf.placeholder(dtype=tf.float32, shape=[None, bbox_padding, 9], name='input_bbox_p')
    return input_coors_p, input_features_p, input_num_list_p, input_bbox_p


def model_stage1(input_coors,
                 input_features,
                 input_num_list,
                 is_training,
                 is_eval,
                 bn):

    base_params = config.base_params
    coors, features, num_list = input_coors, input_features, input_num_list

    with tf.variable_scope("stage1"):
        # =============================== STAGE-1 [base] ================================
        for layer_name in sorted(base_params.keys()):
            coors, features, num_list = point_conv(input_coors=coors,
                                                   input_features=features,
                                                   input_num_list=num_list,
                                                   layer_params=base_params[layer_name],
                                                   scope="stage1_" + layer_name,
                                                   is_training=is_training,
                                                   is_eval=is_eval,
                                                   model_params=model_params,
                                                   bn_decay=bn)

        # =============================== STAGE-1 [rpn] ================================

        roi_coors, roi_features, roi_num_list = point_conv(input_coors=coors,
                                                           input_features=features,
                                                           input_num_list=num_list,
                                                           layer_params=config.rpn_params,
                                                           scope="stage1_rpn_conv",
                                                           is_training=is_training,
                                                           is_eval=is_eval,
                                                           model_params=model_params,
                                                           bn_decay=bn)

        roi_logits = fully_connected(input_points=roi_features,
                                     num_output_channels=config.output_attr,
                                     drop_rate=0.,
                                     model_params=model_params,
                                     scope='stage1_rpn_fc',
                                     is_training=is_training,
                                     last_layer=True)

        roi_attrs = get_roi_attrs(input_logits=roi_logits,
                                  base_coors=roi_coors,
                                  anchor_size=anchor_size,
                                  is_eval=is_eval)

        roi_conf_logits = roi_logits[:, 7]

        return coors, features, num_list, roi_coors, roi_attrs, roi_conf_logits, roi_num_list

def model_stage2(coors,
                 features,
                 num_list,
                 roi_attrs,
                 roi_conf_logits,
                 roi_num_list,
                 is_training,
                 is_eval,
                 bn):
    with tf.variable_scope("stage2"):
        roi_conf = tf.nn.sigmoid(roi_conf_logits)
        bbox_roi_attrs, bbox_num_list, bbox_idx = roi_filter(input_roi_attrs=roi_attrs,
                                                             input_roi_conf=roi_conf,
                                                             input_num_list=roi_num_list,
                                                             conf_thres=config.roi_thres)

        bbox_voxels = roi_pooling(input_coors=coors,
                                  input_features=features,
                                  roi_attrs=bbox_roi_attrs,
                                  input_num_list=num_list,
                                  roi_num_list=bbox_num_list,
                                  voxel_size=config.roi_voxel_size,
                                  pooling_size=5.)

        for i in range(config.roi_voxel_size // 2):
            bbox_voxels = conv_3d(input_voxels=bbox_voxels,
                                  layer_params=config.refine_params,
                                  scope="stage2_refine_conv_{}".format(i),
                                  is_training=is_training,
                                  model_params=model_params,
                                  bn_decay=bn)

        bbox_features = tf.squeeze(bbox_voxels, axis=[1])

        bbox_logits = fully_connected(input_points=bbox_features,
                                      num_output_channels=config.output_attr,
                                      drop_rate=0.,
                                      model_params=model_params,
                                      scope='stage2_refine_fc',
                                      is_training=is_training,
                                      last_layer=True)

        bbox_attrs = get_bbox_attrs(input_logits=bbox_logits,
                                    input_roi_attrs=bbox_roi_attrs,
                                    is_eval=is_eval)

        bbox_conf_logits = bbox_logits[:, 7]

    return bbox_attrs, bbox_conf_logits, bbox_num_list, bbox_idx


def model_test(coors,
                 features,
                 num_list,
                 roi_attrs,
                 roi_conf_logits,
                 roi_num_list,
                 is_training,
                 is_eval,
                 bn):
    with tf.variable_scope("stage2"):
        roi_conf = tf.nn.sigmoid(roi_conf_logits)
        bbox_roi_attrs, bbox_num_list, bbox_idx = roi_filter(input_roi_attrs=roi_attrs,
                                                             input_roi_conf=roi_conf,
                                                             input_num_list=roi_num_list,
                                                             conf_thres=config.roi_thres)

        bbox_voxels = roi_pooling(input_coors=coors,
                                  input_features=features,
                                  roi_attrs=bbox_roi_attrs,
                                  input_num_list=num_list,
                                  roi_num_list=bbox_num_list,
                                  voxel_size=config.roi_voxel_size,
                                  pooling_size=5.)

        # for i in range(config.roi_voxel_size // 2):
        #     bbox_voxels = conv_3d(input_voxels=bbox_voxels,
        #                           layer_params=config.refine_params,
        #                           scope="stage2_refine_conv_{}".format(i),
        #                           is_training=is_training,
        #                           model_params=model_params,
        #                           bn_decay=bn)
        #
        # bbox_features = tf.squeeze(bbox_voxels, axis=[1])
        #
        # bbox_logits = fully_connected(input_points=bbox_features,
        #                               num_output_channels=config.output_attr,
        #                               drop_rate=0.,
        #                               model_params=model_params,
        #                               scope='stage2_refine_fc',
        #                               is_training=is_training,
        #                               last_layer=True)
        #
        # bbox_attrs = get_bbox_attrs(input_logits=bbox_logits,
        #                             input_roi_attrs=bbox_roi_attrs,
        #                             is_eval=is_eval)
        #
        # bbox_conf_logits = bbox_logits[:, 7]

    return bbox_voxels


def loss_stage1(roi_coors,
                pred_roi_attrs,
                roi_conf_logits,
                roi_num_list,
                bbox_labels,
                wd):
    pred_roi_conf = tf.clip_by_value(tf.nn.sigmoid(roi_conf_logits), eps, 1 - eps)
    gt_roi_attrs, gt_roi_conf, gt_roi_diff = get_roi_bbox(input_coors=roi_coors,
                                                          bboxes=bbox_labels,
                                                          input_num_list=roi_num_list,
                                                          anchor_size=anchor_size,
                                                          expand_ratio=0.1,
                                                          diff_thres=2)
    roi_ious = cal_3d_iou(gt_attrs=gt_roi_attrs, pred_attrs=pred_roi_attrs, clip=False)
    roi_iou_masks = tf.cast(tf.equal(gt_roi_conf, 1), dtype=tf.float32) # [-1, 0, 1] -> [0, 0, 1]
    roi_iou_loss = get_masked_average(1. - roi_ious, roi_iou_masks)
    averaged_iou = get_masked_average(roi_ious, roi_iou_masks)

    roi_conf_masks = tf.cast(tf.greater(gt_roi_conf, -1), dtype=tf.float32) # [-1, 0, 1] -> [0, 1, 1]
    roi_conf_target = tf.cast(gt_roi_conf, dtype=tf.float32) * roi_conf_masks # [-1, 0, 1] * [0, 1, 1] -> [0, 0, 1]
    roi_conf_loss = get_masked_average(focal_loss(label=roi_conf_target, pred=pred_roi_conf, alpha=0.25), roi_conf_masks)

    roi_l2_loss = wd * tf.add_n(tf.get_collection("stage1_l2"))

    total_loss = roi_iou_loss + roi_conf_loss + roi_l2_loss
    total_loss_collection = hvd.allreduce(total_loss)

    return total_loss_collection, roi_ious, averaged_iou


def loss_stage2(roi_attrs,
                pred_bbox_attrs,
                bbox_conf_logits,
                bbox_num_list,
                bbox_labels,
                bbox_idx,
                roi_ious,
                wd):
    pred_bbox_conf = tf.clip_by_value(tf.nn.sigmoid(bbox_conf_logits), eps, 1 - eps)
    filtered_roi_attrs = tf.gather(roi_attrs, bbox_idx, axis=0)
    filtered_roi_ious = tf.gather(roi_ious, bbox_idx, axis=0)
    gt_bbox_attrs, gt_bbox_conf, gt_bbox_diff = get_bbox(roi_attrs=filtered_roi_attrs,
                                                         bboxes=bbox_labels,
                                                         input_num_list=bbox_num_list,
                                                         expand_ratio=0.1,
                                                         diff_thres=2)
    bbox_ious = cal_3d_iou(gt_attrs=gt_bbox_attrs, pred_attrs=pred_bbox_attrs, clip=False)
    bbox_iou_masks = tf.cast(tf.equal(gt_bbox_conf, 1), dtype=tf.float32) # [-1, 0, 1] -> [0, 0, 1]
    bbox_iou_loss = get_masked_average(1. - bbox_ious, bbox_iou_masks)
    averaged_iou = get_masked_average(bbox_ious, bbox_iou_masks)

    bbox_conf_masks = tf.cast(tf.greater(gt_bbox_conf, -1), dtype=tf.float32) # [-1, 0, 1] -> [0, 1, 1]
    bbox_conf_target = tf.minimum(tf.maximum(2. * tf.identity(filtered_roi_ious) - 0.5, 0.), 1.) * bbox_conf_masks
    bbox_conf_loss = get_masked_average(-bbox_conf_target * tf.log(pred_bbox_conf) - \
                                        (1 - bbox_conf_target) * tf.log(1 - pred_bbox_conf), bbox_conf_masks)

    bbox_l2_loss = wd * tf.add_n(tf.get_collection("stage2_l2"))

    total_loss = bbox_iou_loss + bbox_conf_loss + bbox_l2_loss
    total_loss_collection = hvd.allreduce(total_loss)
    averaged_iou_collection = hvd.allreduce(averaged_iou)

    return total_loss_collection, averaged_iou_collection

