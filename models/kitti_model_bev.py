import horovod.tensorflow as hvd
import tensorflow as tf
import numpy as np
import train.kitti.kitti_config as config
from models.tf_ops.loader.bbox_utils import get_roi_bbox, get_bbox
from models.tf_ops.loader.others import roi_filter, rotated_nms3d_idx
from models.tf_ops.loader.others import roi_filter, rotated_nms3d_idx
from models.tf_ops.loader.sampling import get_bev_anchor_coors
from models.utils.iou_utils import cal_3d_iou
from models.utils.loss_utils import get_masked_average, focal_loss, smooth_l1_loss, get_dir_cls, get_bbox_loss, get_bbox_from_logits
from models.utils.model_blocks import point_conv, conv_1d, conv_2d, conv_3d, point_conv_res, conv_3d_res, point_conv_concat, bev_compression
from models.utils.layers_wrapper import get_roi_attrs, get_bbox_attrs

anchor_size = config.anchor_size
eps = tf.constant(1e-6)

model_params = {'xavier': config.xavier,
                'stddev': config.stddev,
                'activation': config.activation,
                'padding': -0.5}

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
                        'offset': config.offset_training}


    # base_params = config.base_params_inference if not is_eval else config.base_params_inference
    # rpn_params = config.rpn_params_inference if not is_eval else config.rpn_params_inference

    base_params = config.base_params_inference
    # rpn_params = config.rpn_params_inference

    coors, features, num_list = input_coors, input_features, input_num_list
    concat_features = None
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

        roi_features = concat_features
        roi_coors = coors
        roi_num_list = num_list

        bev_img = bev_compression(input_coors=roi_coors,
                                  input_features=roi_features,
                                  input_num_list=roi_num_list,
                                  resolution=[0.6, 0.6, 0.8],
                                  dimension_params=dimension_params)

        print(bev_img)

        bev_img = conv_2d(input_img=bev_img,
                          kernel_size=3,
                          num_output_channels=128,
                          model_params=model_params,
                          scope='stage1_rpn_conv2d_0',
                          is_training=is_training,
                          trainable=trainable)

        bev_img = conv_2d(input_img=bev_img,
                          kernel_size=3,
                          num_output_channels=128,
                          model_params=model_params,
                          scope='stage1_rpn_conv2d_1',
                          is_training=is_training,
                          trainable=trainable,
                          second_last_layer=True)

        logits_img = conv_2d(input_img=bev_img,
                             kernel_size=1,
                             num_output_channels=2 * config.output_attr,
                             model_params=model_params,
                             scope='stage1_rpn_conv2d_2',
                             is_training=is_training,
                             trainable=trainable,
                             last_layer=True)  # [n, w, l, 2*8] -> [n*w*l*2, 8]

        roi_logits = tf.reshape(logits_img, shape=[-1, config.output_attr])

        anchor_coors, anchor_num_list = get_bev_anchor_coors(bev_img=logits_img,
                                                       resolution=[0.6, 0.6, 0.8],
                                                       offset=dimension_params['offset'],
                                                       height=-1.)
        #
        # roi_attrs = get_bbox_from_logits(point_coors=roi_coors,
        #                                  pred_logits=roi_logits,
        #                                  anchor_size=config.anchor_size)

        return coors, features, num_list, anchor_coors, roi_logits, anchor_num_list


def stage1_loss(anchor_coors,
                roi_logits,
                anchor_num_list,
                bbox_labels,
                wd):
    pred_roi_conf = tf.clip_by_value(tf.nn.sigmoid(roi_logits[:, -1]), eps, 1 - eps)
    gt_roi_attrs_0, gt_roi_conf, gt_roi_diff = get_roi_bbox(input_coors=anchor_coors,
                                                            bboxes=bbox_labels,
                                                            input_num_list=anchor_num_list,
                                                            anchor_size=anchor_size,
                                                            expand_ratio=0.1,
                                                            diff_thres=2,
                                                            anchor_rot=[0, np.pi/2])
    gt_roi_attrs_1 = get_90_rotated_attrs(gt_roi_attrs_0)
    gt_roi_attrs = tf.stack([gt_roi_attrs_0, gt_roi_attrs_1], axis=1) # [n, 2, 7]
    gt_anchor_bool_masks = tf.greater(tf.abs(tf.sin(gt_roi_attrs_0[:, 6])), tf.abs(tf.sin(gt_roi_attrs_1[:, 6])))
    gt_anchor_cls = tf.cast(gt_anchor_bool_masks, dtype=tf.int32)
    gt_roi_attrs = tf.gather_nd(params=gt_roi_attrs,
                                indices=tf.stack([tf.range(tf.shape(gt_anchor_cls)[0]), gt_anchor_cls], axis=-1))

    roi_ious = cal_3d_iou(gt_attrs=gt_roi_attrs, pred_attrs=pred_roi_attrs, clip=False)
    roi_iou_masks = tf.cast(tf.equal(gt_roi_conf, 1), dtype=tf.float32) # [-1, 0, 1] -> [0, 0, 1]
    roi_iou_loss = get_masked_average(1. - roi_ious, roi_iou_masks)
    tf.summary.scalar('stage1_iou_loss', roi_iou_loss)
    averaged_iou = get_masked_average(roi_ious, roi_iou_masks)

    roi_l1_loss = smooth_l1_loss(predictions=pred_roi_attrs[:, 6], labels=gt_roi_attrs[:, 6])
    roi_l1_loss = get_masked_average(roi_l1_loss, roi_iou_masks)
    tf.summary.scalar('stage1_l1_loss', roi_l1_loss)
    tf.summary.scalar('roi_angle_sin_bias', get_masked_average(tf.abs(tf.sin(gt_roi_attrs[:, 6] - pred_roi_attrs[:, 6])), roi_iou_masks))
    tf.summary.scalar('roi_angle_bias', get_masked_average(tf.abs(gt_roi_attrs[:, 6] - pred_roi_attrs[:, 6]), roi_iou_masks))

    roi_conf_masks = tf.cast(tf.greater(gt_roi_conf, -1), dtype=tf.float32) # [-1, 0, 1] -> [0, 1, 1]
    roi_conf_target = tf.cast(gt_roi_conf, dtype=tf.float32) * roi_conf_masks # [-1, 0, 1] * [0, 1, 1] -> [0, 0, 1]
    roi_conf_loss = get_masked_average(focal_loss(label=roi_conf_target, pred=pred_roi_conf, alpha=0.25), roi_conf_masks)
    tf.summary.scalar('stage1_conf_loss', roi_conf_loss)

    gt_anchor_cls = tf.clip_by_value(tf.cast(gt_anchor_cls, dtype=tf.float32), eps, 1. - eps)
    anchor_cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_anchor_cls, logits=anchor_cls_logits)
    anchor_cls_loss = get_masked_average(anchor_cls_loss, roi_iou_masks)
    tf.summary.scalar('anchor_cls_loss', anchor_cls_loss)

    roi_l2_loss = wd * tf.add_n(tf.get_collection("stage1_l2"))
    tf.summary.scalar('stage1_l2_loss', roi_l2_loss)

    # total_loss = roi_iou_loss + roi_l1_loss + roi_conf_loss + roi_l2_loss
    total_loss = roi_iou_loss + anchor_cls_loss + roi_l1_loss + roi_conf_loss + roi_l2_loss
    total_loss_collection = hvd.allreduce(total_loss)

    return total_loss_collection, roi_ious, averaged_iou