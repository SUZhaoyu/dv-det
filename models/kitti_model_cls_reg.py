import horovod.tensorflow as hvd
import tensorflow as tf

import train.kitti.kitti_config as config
from models.tf_ops.loader.bbox_utils import get_roi_bbox, get_bbox
from models.tf_ops.loader.others import roi_filter, rotated_nms3d_idx
from models.tf_ops.loader.pooling import la_roi_pooling_fast
from models.utils.iou_utils import cal_3d_iou
from models.utils.loss_utils import get_masked_average, focal_loss, smooth_l1_loss, get_dir_cls, get_bbox_loss, get_bbox_from_logits
from models.utils.model_blocks import point_conv, conv_1d, conv_3d, point_conv_res, conv_3d_res, point_conv_concat
from models.utils.layers_wrapper import get_roi_attrs, get_bbox_attrs

anchor_size = config.anchor_size
eps = tf.constant(1e-6)

model_params = {'xavier': config.xavier,
                'stddev': config.stddev,
                'activation': config.activation,
                'padding': 0.,}

def stage1_inputs_placeholder(input_channels=1,
                              bbox_padding=config.aug_config['nbbox']):
    input_coors_p = tf.placeholder(tf.float32, shape=[None, 3], name='stage1_input_coors_p')
    input_features_p = tf.placeholder(tf.float32, shape=[None, input_channels], name='stage1_input_features_p')
    input_num_list_p = tf.placeholder(tf.int32, shape=[None], name='stage1_input_num_list_p')
    input_bbox_p = tf.placeholder(dtype=tf.float32, shape=[None, bbox_padding, 9], name='stage1_input_bbox_p')
    return input_coors_p, input_features_p, input_num_list_p, input_bbox_p

# config.base_params_inference[sorted(config.base_params_inference.keys())[-1]]['c_out']
def stage2_inputs_placeholder(input_feature_channels=385,
                              bbox_padding=config.aug_config['nbbox']):
    input_coors_p = tf.placeholder(tf.float32, shape=[None, 3], name='stage2_input_coors_p')
    input_features_p = tf.placeholder(tf.float32, shape=[None, input_feature_channels], name='stage2_input_features_p')
    input_num_list_p = tf.placeholder(tf.int32, shape=[None], name='stage2_input_num_list_p')
    input_roi_coors_p = tf.placeholder(tf.float32, shape=[None, 3], name='stage_input_roi_coors_p')
    input_roi_attrs_p = tf.placeholder(tf.float32, shape=[None, 7], name='stage2_input_roi_attrs_p')
    input_roi_conf_p = tf.placeholder(tf.float32, shape=[None], name='stage2_input_roi_conf_p')
    input_roi_num_list_p = tf.placeholder(tf.int32, shape=[None], name='stage2_input_roi_num_list_p')
    input_bbox_p = tf.placeholder(dtype=tf.float32, shape=[None, bbox_padding, 9], name='stage2_input_bbox_p')
    return input_coors_p, input_features_p, input_num_list_p, input_roi_coors_p, \
           input_roi_attrs_p, input_roi_conf_p, input_roi_num_list_p, input_bbox_p


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
                        'offset': config.offset_training}


    # base_params = config.base_params_inference if not is_eval else config.base_params_inference
    # rpn_params = config.rpn_params_inference if not is_eval else config.rpn_params_inference

    base_params = config.base_params_inference
    # rpn_params = config.rpn_params_inference

    coors, features, num_list = input_coors, input_features, input_num_list
    concat_features = features
    voxel_idx, center_idx = None, None

    with tf.variable_scope("stage1"):
        # =============================== STAGE-1 [base] ================================

        for i, layer_name in enumerate(sorted(base_params.keys())):
            coors, features, num_list, voxel_idx, center_idx, concat_features = \
                point_conv_concat(input_coors=coors,
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

        # for i, layer_name in enumerate(sorted(rpn_params.keys())):
        #     roi_coors, roi_features, roi_num_list, _, _ = \
        #         point_conv(input_coors=coors,
        #                    input_features=features,
        #                    input_num_list=num_list,
        #                    voxel_idx=voxel_idx,
        #                    center_idx=center_idx,
        #                    layer_params=rpn_params[layer_name],
        #                    dimension_params=dimension_params,
        #                    grid_buffer_size=config.grid_buffer_size,
        #                    output_pooling_size=config.output_pooling_size,
        #                    scope="stage1_" + layer_name,
        #                    is_training=is_training,
        #                    trainable=trainable,
        #                    mem_saving=mem_saving,
        #                    model_params=model_params,
        #                    bn_decay=bn)

        roi_features = concat_features
        roi_coors = coors
        roi_num_list = num_list

        # roi_features = conv_1d(input_points=roi_features,
        #                        num_output_channels=256,
        #                        drop_rate=0.,
        #                        model_params=model_params,
        #                        scope='stage1_rpn_fc_0',
        #                        is_training=is_training,
        #                        trainable=trainable)

        roi_features = conv_1d(input_points=roi_features,
                               num_output_channels=256,
                               drop_rate=0.,
                               model_params=model_params,
                               scope='stage1_rpn_fc_1',
                               is_training=is_training,
                               trainable=trainable,
                               second_last_layer=True)

        roi_logits = conv_1d(input_points=roi_features,
                             num_output_channels=32 + 1,
                             drop_rate=0.,
                             model_params=model_params,
                             scope='stage1_rpn_fc_2',
                             is_training=is_training,
                             trainable=trainable,
                             last_layer=True)

        # roi_attrs = get_roi_attrs(input_logits=roi_logits,
        #                           base_coors=roi_coors,
        #                           anchor_size=anchor_size,
        #                           is_eval=is_eval)

        roi_conf_logits = roi_logits[:, -1]

        # roi_conf = tf.nn.sigmoid(roi_conf_logits)
        # nms_idx = rotated_nms3d_idx(roi_attrs, roi_conf, nms_overlap_thresh=0.7, nms_conf_thres=0.3)
        # roi_coors = tf.gather(roi_coors, nms_idx, axis=0)
        # roi_attrs = tf.gather(roi_attrs, nms_idx, axis=0)
        # roi_conf_logits = tf.gather(roi_conf_logits, nms_idx, axis=0)
        # roi_num_list = tf.expand_dims(tf.shape(nms_idx)[0], axis=0)

        return coors, concat_features, num_list, roi_coors, roi_logits, roi_conf_logits, roi_num_list



def stage1_loss(roi_coors,
                pred_roi_logits,
                roi_conf_logits,
                roi_num_list,
                bbox_labels,
                wd):
    pred_roi_conf = tf.clip_by_value(tf.nn.sigmoid(roi_conf_logits), eps, 1 - eps)
    gt_roi_attrs, gt_roi_conf, gt_roi_diff = get_roi_bbox(input_coors=roi_coors,
                                                          bboxes=bbox_labels,
                                                          input_num_list=roi_num_list,
                                                          anchor_size=anchor_size,
                                                          expand_ratio=0.2,
                                                          diff_thres=config.diff_thres,
                                                          cls_thres=config.cls_thres)
    # gt_roi_logits = roi_attrs_to_logits(roi_coors, gt_roi_attrs, anchor_size)
    # pred_roi_logits = roi_attrs_to_logits(roi_coors, pred_roi_attrs, anchor_size)
    # gt_roi_attrs = roi_logits_to_attrs_tf(roi_coors, gt_roi_logits, anchor_size)
    # pred_roi_attrs = roi_logits_to_attrs_tf(roi_coors, pred_roi_logits, anchor_size)

    pred_roi_attrs = get_bbox_from_logits(point_coors=roi_coors,
                                          pred_logits=pred_roi_logits,
                                          anchor_size=anchor_size,
                                          )
    roi_ious = cal_3d_iou(gt_attrs=gt_roi_attrs, pred_attrs=pred_roi_attrs, clip=False)
    roi_iou_masks = tf.cast(tf.equal(gt_roi_conf, 1), dtype=tf.float32)  # [-1, 0, 1] -> [0, 0, 1]
    averaged_iou = get_masked_average(roi_ious, roi_iou_masks)

    roi_loss = get_bbox_loss(point_coors=roi_coors,
                             pred_logits=pred_roi_logits,
                             label_attrs=gt_roi_attrs,
                             foreground_masks=roi_iou_masks,
                             anchor_size=anchor_size)
    tf.summary.scalar('stage1_roi_loss', roi_loss)


    roi_conf_masks = tf.cast(tf.greater(gt_roi_conf, -1), dtype=tf.float32) # [-1, 0, 1] -> [0, 1, 1]
    roi_conf_target = tf.cast(gt_roi_conf, dtype=tf.float32) * roi_conf_masks # [-1, 0, 1] * [0, 1, 1] -> [0, 0, 1]
    roi_conf_loss = get_masked_average(focal_loss(label=roi_conf_target, pred=pred_roi_conf, alpha=0.75), roi_conf_masks)
    tf.summary.scalar('stage1_conf_loss', roi_conf_loss)

    roi_l2_loss = wd * tf.add_n(tf.get_collection("stage1_l2"))
    tf.summary.scalar('stage1_l2_loss', roi_l2_loss)

    total_loss = roi_loss + roi_conf_loss + roi_l2_loss
    # total_loss_collection = hvd.allreduce(total_loss)
    averaged_iou_collection = hvd.allreduce(averaged_iou)

    return total_loss, averaged_iou_collection


def get_roi_iou(roi_coors, pred_roi_attrs, roi_num_list, bbox_labels):
    gt_roi_attrs, gt_roi_conf, gt_roi_diff = get_roi_bbox(input_coors=roi_coors,
                                                          bboxes=bbox_labels,
                                                          input_num_list=roi_num_list,
                                                          anchor_size=anchor_size,
                                                          expand_ratio=0.2,
                                                          diff_thres=config.diff_thres,
                                                          cls_thres=config.cls_thres)
    roi_ious = cal_3d_iou(gt_attrs=gt_roi_attrs, pred_attrs=pred_roi_attrs, clip=False)
    return roi_ious

