import horovod.tensorflow as hvd
import tensorflow as tf
import numpy as np

import train.kitti.kitti_config as config
from models.tf_ops.loader.bbox_utils import get_roi_bbox, get_bbox, logits_to_attrs, get_anchor_attrs
from models.tf_ops.loader.others import anchor_iou_filter
from models.tf_ops.loader.pooling import la_roi_pooling_fast
from models.tf_ops.loader.sampling import get_bev_features
from models.utils.iou_utils import cal_3d_iou, cal_bev_iou
from models.utils.loss_utils import get_masked_average, focal_loss, smooth_l1_loss, get_dir_cls
from models.utils.model_blocks import point_conv, conv_1d, conv_3d, point_conv_res, conv_3d_res, point_conv_bev_concat
from models.tf_ops.loader.bbox_utils import get_roi_bbox, get_anchor_attrs, get_bev_gt_bbox
from models.utils.layers_wrapper import get_roi_attrs, get_bbox_attrs

anchor_size = config.anchor_size
eps = tf.constant(1e-6)

anchor_param_list = tf.constant([[1.6, 3.9, 1.5, -1.0, 0],
                                 [1.6, 3.9, 1.5, -1.0, np.pi / 2]])

model_params = {'xavier': config.xavier,
                'stddev': config.stddev,
                'activation': config.activation,
                'padding': 0.,
                'bev_channels': 128,
                'bev_resolution': 0.4}

def stage1_inputs_placeholder(input_channels=1,
                              bbox_padding=config.aug_config['nbbox']):
    input_coors_p = tf.placeholder(tf.float32, shape=[None, 3], name='stage1_input_coors_p')
    input_features_p = tf.placeholder(tf.float32, shape=[None, input_channels], name='stage1_input_features_p')
    input_num_list_p = tf.placeholder(tf.int32, shape=[None], name='stage1_input_num_list_p')
    input_bbox_p = tf.placeholder(dtype=tf.float32, shape=[None, bbox_padding, 9], name='stage1_input_bbox_p')
    return input_coors_p, input_features_p, input_num_list_p, input_bbox_p


def stage1_model(input_coors,
                 input_features,
                 input_num_list,
                 is_training,
                 trainable,
                 is_eval,
                 mem_saving,
                 bn):
    # if is_eval:
    #     dimension_params = {'dimension': config.dimension_testing,
    #                         'offset': config.offset_testing}
    # else:
    dimension_params = {'dimension': config.dimension_training,
                        'offset': config.offset_training,
                        'bev_size': [config.batch_size,
                                     int(np.ceil((config.dimension_training[0]/model_params['bev_resolution']))),
                                     int(np.ceil((config.dimension_training[1]/model_params['bev_resolution']))),
                                     model_params['bev_channels']]}


    # base_params = config.base_params_inference if not is_eval else config.base_params_inference
    # rpn_params = config.rpn_params_inference if not is_eval else config.rpn_params_inference

    base_params = config.base_params_inference
    # rpn_params = config.rpn_params_inference

    coors, features, num_list = input_coors, input_features, input_num_list
    concat_features = []
    voxel_idx, center_idx = None, None

    with tf.variable_scope("stage1"):
        # =============================== STAGE-1 [base] ================================

        for i, layer_name in enumerate(sorted(base_params.keys())):
            coors, features, num_list, voxel_idx, center_idx, concat_features = \
                point_conv_bev_concat(input_coors=coors,
                                      input_features=features,
                                      concat_features=concat_features,
                                      input_num_list=num_list,
                                      voxel_idx=voxel_idx,
                                      center_idx=center_idx,
                                      layer_params=base_params[layer_name],
                                      dimension_params=dimension_params,
                                      grid_buffer_size=config.grid_buffer_size,
                                      output_pooling_size=config.output_pooling_size,
                                      scope="stage1_" + layer_name,
                                      is_training=is_training,
                                      trainable=trainable,
                                      mem_saving=mem_saving,
                                      model_params=model_params,
                                      bn_decay=bn)


        # =============================== STAGE-1 [rpn] ================================

        bev_features = tf.concat(concat_features, axis=-1)
        bev_coors, bbox_features, bbox_num_list = get_bev_features(bev_img=bev_features,
                                                                   resolution=model_params['bev_resolution'],
                                                                   offset=dimension_params['offset'])

        # roi_features = conv_1d(input_points=roi_features,
        #                        num_output_channels=256,
        #                        drop_rate=0.,
        #                        model_params=model_params,
        #                        scope='stage1_rpn_fc_0',
        #                        is_training=is_training,
        #                        trainable=trainable)
        #
        # roi_features = conv_1d(input_points=roi_features,
        #                        num_output_channels=256,
        #                        drop_rate=0.,
        #                        model_params=model_params,
        #                        scope='stage1_rpn_fc_1',
        #                        is_training=is_training,
        #                        trainable=trainable,
        #                        second_last_layer=True)

        logits = conv_1d(input_points=bbox_features,
                         num_output_channels=config.output_attr * 2,
                         drop_rate=0.,
                         model_params=model_params,
                         scope='stage1_rpn_fc_2',
                         is_training=is_training,
                         trainable=trainable,
                         last_layer=True)
        logits = tf.stack(tf.split(logits, 2, axis=1), axis=1) # [n, 2, f]

        bbox_attrs = logits_to_attrs(anchor_coors=bev_coors,
                                     input_logits=logits,
                                     anchor_param_list=anchor_param_list) # [n, 2, f]

        conf_logits = logits[..., 7]  # [n, 2]

        return bev_coors, bbox_attrs, conf_logits, bbox_num_list


def stage1_loss(bev_coors,
                pred_attrs,
                conf_logits,
                num_list,
                bbox_labels,
                wd):
    pred_conf = tf.nn.sigmoid(conf_logits)
    gt_attrs, gt_conf, gt_idx = get_bev_gt_bbox(input_coors=bev_coors,
                                                label_bbox=bbox_labels,
                                                input_num_list=num_list,
                                                anchor_param_list=anchor_param_list,
                                                expand_ratio=0.15,
                                                diff_thres=4,
                                                cls_thres=1)

    anchor_attrs = get_anchor_attrs(anchor_coors=bev_coors,
                                    anchor_param_list=anchor_param_list)
    # print(anchor_attrs, pred_attrs, gt_attrs)

    gt_attrs = tf.reshape(gt_attrs, [-1, tf.shape(gt_attrs)[2]]) # all the anchor locations
    anchor_attrs = tf.reshape(anchor_attrs, [-1, tf.shape(anchor_attrs)[2]])
    pred_attrs = tf.reshape(pred_attrs, [-1, tf.shape(pred_attrs)[2]])
    gt_conf = tf.reshape(gt_conf, [-1])
    gt_idx = tf.reshape(gt_idx, [-1])
    pred_conf = tf.reshape(pred_conf, [-1])
    # print(anchor_attrs, pred_attrs, gt_attrs)

    positive_idx = tf.where(tf.equal(gt_conf, 1))[:, 0] # only select the anchors at object locations
    positive_gt_attrs = tf.gather(gt_attrs, positive_idx, axis=0)
    positive_gt_idx = tf.gather(gt_idx, positive_idx)
    positive_anchor_attrs = tf.gather(anchor_attrs, positive_idx, axis=0)
    positive_pred_attrs = tf.gather(pred_attrs, positive_idx, axis=0)

    # print(tf.shape(pred_attrs), tf.shape(gt_attrs))

    positive_bev_iou = cal_bev_iou(positive_gt_attrs, positive_anchor_attrs) # calculate bev ious
    gt_conf = anchor_iou_filter(positive_bev_iou, positive_gt_idx, bbox_labels, gt_conf, positive_idx) # classify all the gt_conf into [-1, 0, 1]
    positive_gt_conf = tf.gather(gt_conf, positive_idx)

    conf_masks = tf.cast(tf.greater(gt_conf, -1), dtype=tf.float32)  # [-1, 0, 1] -> [0, 1, 1]
    conf_target = tf.cast(gt_conf, dtype=tf.float32) * conf_masks  # [-1, 0, 1] * [0, 1, 1] -> [0, 0, 1]
    conf_loss = get_masked_average(focal_loss(label=conf_target, pred=pred_conf, alpha=0.75), conf_masks)
    tf.summary.scalar('conf_loss', conf_loss)

    print(positive_gt_attrs, positive_pred_attrs)
    positive_pred_ious = cal_3d_iou(gt_attrs=positive_gt_attrs, pred_attrs=positive_pred_attrs, clip=False)
    positive_iou_masks = tf.cast(tf.equal(positive_gt_conf, 1), dtype=tf.float32)  # [-1, 0, 1] -> [0, 0, 1]
    iou_loss = get_masked_average(1. - positive_pred_ious, positive_iou_masks)
    tf.summary.scalar('iou_loss', iou_loss)
    averaged_iou = get_masked_average(positive_pred_ious, positive_iou_masks)

    l1_loss = smooth_l1_loss(predictions=positive_pred_attrs[:, 6], labels=positive_gt_attrs[:, 6], delta=1. / 9.)
    l1_loss = get_masked_average(l1_loss, positive_iou_masks)
    tf.summary.scalar('l1_loss', l1_loss)
    tf.summary.scalar('angle_sin_bias',
                      get_masked_average(tf.abs(tf.sin(positive_gt_attrs[:, 6] - positive_pred_attrs[:, 6])), positive_iou_masks))
    # tf.summary.scalar('angle_bias',
    #                   get_masked_average(tf.abs(positive_gt_attrs[:, 6] - positive_pred_attrs[:, 6]), positive_iou_masks))

    l2_loss = wd * tf.add_n(tf.get_collection("stage1_l2"))
    tf.summary.scalar('l2_loss', l2_loss)

    total_loss = iou_loss + l1_loss + conf_loss + l2_loss
    total_loss_collection = hvd.allreduce(total_loss)
    averaged_iou_collection = hvd.allreduce(averaged_iou)

    return total_loss_collection, averaged_iou_collection