import horovod.tensorflow as hvd
import tensorflow as tf

import train.waymo.waymo_config as config
from models.tf_ops.loader.bbox_utils import get_roi_bbox, get_bbox
from models.tf_ops.loader.others import roi_filter, iou_filtering
from models.tf_ops.loader.pooling import la_roi_pooling_fast
from models.utils.iou_utils import cal_3d_iou
from models.utils.loss_utils import get_masked_average, focal_loss, smooth_l1_loss, get_dir_cls
from models.utils.model_layers import point_conv, fully_connected, conv_3d, point_conv_concat
from models.utils.ops_wrapper import get_roi_attrs, get_bbox_attrs

anchor_size = config.anchor_size
eps = tf.constant(1e-6)

model_params = {'xavier': config.xavier,
                'stddev': config.stddev,
                'activation': config.activation}

def stage1_inputs_placeholder(input_channels=1,
                              bbox_padding=config.aug_config['nbbox']):
    input_coors_p = tf.placeholder(tf.float32, shape=[None, 3], name='stage1_input_coors_p')
    input_features_p = tf.placeholder(tf.float32, shape=[None, input_channels], name='stage1_input_features_p')
    input_num_list_p = tf.placeholder(tf.int32, shape=[None], name='stage1_input_num_list_p')
    input_bbox_p = tf.placeholder(dtype=tf.float32, shape=[None, bbox_padding, 9], name='stage1_input_bbox_p')
    return input_coors_p, input_features_p, input_num_list_p, input_bbox_p


def stage2_inputs_placeholder(input_feature_channels=config.stage2_input_channels,
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
    rpn_params = config.rpn_params_inference

    coors, features, num_list = input_coors, input_features, input_num_list
    concat_features = features
    voxel_idx, center_idx = None, None

    with tf.variable_scope("stage1"):
        # =============================== STAGE-1 [base] ================================
        for layer_name in sorted(base_params.keys()):
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

        roi_coors, roi_features, roi_num_list, _, _ = \
            point_conv(input_coors=coors,
                       input_features=features,
                       input_num_list=num_list,
                       voxel_idx=voxel_idx,
                       center_idx=center_idx,
                       layer_params=rpn_params,
                       dimension_params=dimension_params,
                       grid_buffer_size=config.grid_buffer_size,
                       output_pooling_size=config.output_pooling_size,
                       scope="stage1_rpn_conv",
                       is_training=is_training,
                       trainable=trainable,
                       mem_saving=mem_saving,
                       model_params=model_params,
                       bn_decay=bn)

        roi_logits = fully_connected(input_points=roi_features,
                                     num_output_channels=config.output_attr,
                                     drop_rate=0.,
                                     model_params=model_params,
                                     scope='stage1_rpn_fc',
                                     is_training=is_training,
                                     trainable=trainable,
                                     last_layer=True)

        roi_attrs = get_roi_attrs(input_logits=roi_logits,
                                  base_coors=roi_coors,
                                  anchor_size=anchor_size,
                                  is_eval=is_eval)

        roi_conf_logits = roi_logits[:, 7]

        if not trainable:
            roi_attrs, roi_coors, roi_conf_logits, roi_num_list = \
                iou_filtering(attrs=roi_attrs,
                              coors=roi_coors,
                              conf_logits=roi_conf_logits,
                              num_list=roi_num_list,
                              nms_overlap_thresh=0.8,
                              nms_conf_thres=config.roi_thres,
                              offset=config.offset_training)


        return coors, concat_features, num_list, roi_coors, roi_attrs, roi_conf_logits, roi_num_list

def stage2_model(coors,
                 features,
                 num_list,
                 roi_attrs,
                 roi_conf_logits,
                 roi_ious,
                 roi_num_list,
                 is_training,
                 trainable,
                 is_eval,
                 mem_saving,
                 bn):

    roi_conf = tf.nn.sigmoid(roi_conf_logits)
    with tf.variable_scope("stage2"):
        bbox_roi_attrs, bbox_num_list, bbox_idx = roi_filter(input_roi_attrs=roi_attrs,
                                                             input_roi_conf=roi_conf,
                                                             input_roi_ious=roi_ious,
                                                             input_num_list=roi_num_list,
                                                             conf_thres=config.roi_thres,
                                                             iou_thres=config.iou_thres,
                                                             max_length=config.max_length,
                                                             with_negative=False)

        bbox_voxels = la_roi_pooling_fast(input_coors=coors,
                                          input_features=features,
                                          roi_attrs=bbox_roi_attrs,
                                          input_num_list=num_list,
                                          roi_num_list=bbox_num_list,
                                          voxel_size=config.roi_voxel_size,
                                          grid_buffer_size=16,
                                          grid_buffer_resolution=2.,
                                          pooling_size=8,
                                          dimension=config.dimension_training,
                                          offset=config.offset_training)

        for i in range(config.roi_voxel_size // 2):
            bbox_voxels = conv_3d(input_voxels=bbox_voxels,
                                  layer_params=config.refine_params,
                                  scope="stage2_refine_conv_{}".format(i),
                                  is_training=is_training,
                                  trainable=trainable,
                                  model_params=model_params,
                                  mem_saving=mem_saving,
                                  bn_decay=bn)

        bbox_features = tf.squeeze(bbox_voxels, axis=[1])

        bbox_logits = fully_connected(input_points=bbox_features,
                                      num_output_channels=config.output_attr + 1,
                                      drop_rate=0.,
                                      model_params=model_params,
                                      scope='stage2_refine_fc',
                                      is_training=is_training,
                                      trainable=trainable,
                                      last_layer=True)

        bbox_attrs = get_bbox_attrs(input_logits=bbox_logits,
                                    input_roi_attrs=bbox_roi_attrs,
                                    is_eval=is_eval)

        bbox_conf_logits = bbox_logits[:, 7]
        bbox_dir_logits = bbox_logits[:, 8]

    return bbox_attrs, bbox_conf_logits, bbox_dir_logits, bbox_num_list, bbox_idx


def stage1_loss(roi_coors,
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
                                                          expand_ratio=0.2,
                                                          diff_thres=4)
    # gt_roi_logits = roi_attrs_to_logits(roi_coors, gt_roi_attrs, anchor_size)
    # pred_roi_logits = roi_attrs_to_logits(roi_coors, pred_roi_attrs, anchor_size)
    # gt_roi_attrs = roi_logits_to_attrs_tf(roi_coors, gt_roi_logits, anchor_size)
    # pred_roi_attrs = roi_logits_to_attrs_tf(roi_coors, pred_roi_logits, anchor_size)

    roi_ious = cal_3d_iou(gt_attrs=gt_roi_attrs, pred_attrs=pred_roi_attrs, clip=False)
    roi_iou_masks = tf.cast(tf.equal(gt_roi_conf, 1), dtype=tf.float32) # [-1, 0, 1] -> [0, 0, 1]
    roi_iou_loss = get_masked_average(1. - roi_ious, roi_iou_masks)
    tf.summary.scalar('stage1_iou_loss', roi_iou_loss)
    averaged_iou = get_masked_average(roi_ious, roi_iou_masks)

    roi_l1_loss = smooth_l1_loss(predictions=pred_roi_attrs[:, 6], labels=gt_roi_attrs[:, 6], delta=1./9.)
    roi_l1_loss = get_masked_average(roi_l1_loss, roi_iou_masks)
    tf.summary.scalar('stage1_l1_loss', roi_l1_loss)
    tf.summary.scalar('roi_angle_sin_bias', get_masked_average(tf.abs(tf.sin(gt_roi_attrs[:, 6] - pred_roi_attrs[:, 6])), roi_iou_masks))
    tf.summary.scalar('roi_angle_bias', get_masked_average(tf.abs(gt_roi_attrs[:, 6] - pred_roi_attrs[:, 6]), roi_iou_masks))

    roi_conf_masks = tf.cast(tf.greater(gt_roi_conf, -1), dtype=tf.float32) # [-1, 0, 1] -> [0, 1, 1]
    roi_conf_target = tf.cast(gt_roi_conf, dtype=tf.float32) * roi_conf_masks # [-1, 0, 1] * [0, 1, 1] -> [0, 0, 1]
    roi_conf_loss = get_masked_average(focal_loss(label=roi_conf_target, pred=pred_roi_conf, alpha=0.75), roi_conf_masks)
    tf.summary.scalar('stage1_conf_loss', roi_conf_loss)

    roi_l2_loss = wd * tf.add_n(tf.get_collection("stage1_l2"))
    tf.summary.scalar('stage1_l2_loss', roi_l2_loss)

    total_loss = roi_iou_loss + roi_l1_loss + roi_conf_loss + roi_l2_loss
    # total_loss = roi_iou_loss + roi_conf_loss + roi_l2_loss
    total_loss_collection = hvd.allreduce(total_loss)
    averaged_iou_collection = hvd.allreduce(averaged_iou)

    return total_loss_collection, averaged_iou_collection

def get_roi_iou(roi_coors, pred_roi_attrs, roi_num_list, bbox_labels, clip=False):
    gt_roi_attrs, gt_roi_conf, gt_roi_diff = get_roi_bbox(input_coors=roi_coors,
                                                          bboxes=bbox_labels,
                                                          input_num_list=roi_num_list,
                                                          anchor_size=anchor_size,
                                                          expand_ratio=0.2,
                                                          diff_thres=4)
    roi_ious = cal_3d_iou(gt_attrs=gt_roi_attrs, pred_attrs=pred_roi_attrs, clip=clip)
    return roi_ious


def stage2_loss(roi_attrs,
                pred_bbox_attrs,
                bbox_conf_logits,
                bbox_dir_logits,
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
                                                         expand_ratio=0.2,
                                                         diff_thres=4)
    bbox_ious = cal_3d_iou(gt_attrs=gt_bbox_attrs, pred_attrs=pred_bbox_attrs, clip=False)
    bbox_iou_masks = tf.cast(tf.logical_and(tf.equal(gt_bbox_conf, 1), tf.greater(filtered_roi_ious, 0.25)), dtype=tf.float32) # [-1, 0, 1] -> [0, 0, 1]
    bbox_iou_loss = get_masked_average(1. - bbox_ious, bbox_iou_masks)
    averaged_iou = get_masked_average(bbox_ious, bbox_iou_masks)
    tf.summary.scalar('stage2_iou_loss', bbox_iou_loss)

    bbox_l1_loss = smooth_l1_loss(predictions=pred_bbox_attrs[:, 6], labels=gt_bbox_attrs[:, 6], delta=1./9.)
    bbox_l1_loss = get_masked_average(bbox_l1_loss, bbox_iou_masks)
    tf.summary.scalar('stage2_l1_loss', bbox_l1_loss)
    tf.summary.scalar('bbox_angle_sin_bias', get_masked_average(tf.abs(tf.sin(gt_bbox_attrs[:, 6] - pred_bbox_attrs[:, 6])), bbox_iou_masks))
    tf.summary.scalar('bbox_angle_bias', get_masked_average(tf.abs(gt_bbox_attrs[:, 6] - pred_bbox_attrs[:, 6]), bbox_iou_masks))

    bbox_dir_cls = get_dir_cls(label=gt_bbox_attrs[:, 6], pred=filtered_roi_attrs[:, 6])
    bbox_dir_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=bbox_dir_cls, logits=bbox_dir_logits)
    bbox_dir_loss = get_masked_average(bbox_dir_loss, bbox_iou_masks)
    tf.summary.scalar('stage2_dir_loss', bbox_dir_loss)

    bbox_conf_masks = tf.cast(tf.greater(gt_bbox_conf, -1), dtype=tf.float32) # [-1, 0, 1] -> [0, 1, 1]
    bbox_conf_target = tf.minimum(tf.maximum(2. * tf.identity(filtered_roi_ious) - 0.5, 0.), 1.) * bbox_conf_masks
    bbox_conf_loss = get_masked_average(-bbox_conf_target * tf.log(pred_bbox_conf) - \
                                        (1 - bbox_conf_target) * tf.log(1 - pred_bbox_conf), bbox_conf_masks)
    tf.summary.scalar('stage2_conf_loss', bbox_conf_loss)

    bbox_l2_loss = wd * tf.add_n(tf.get_collection("stage2_l2"))
    tf.summary.scalar('stage2_l2_loss', bbox_l2_loss)

    total_loss = bbox_iou_loss + bbox_l1_loss + bbox_conf_loss + bbox_dir_loss + bbox_l2_loss
    # total_loss = bbox_iou_loss + bbox_conf_loss + bbox_l2_loss
    total_loss_collection = hvd.allreduce(total_loss)
    averaged_iou_collection = hvd.allreduce(averaged_iou)

    return total_loss_collection, averaged_iou_collection

