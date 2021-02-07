from __future__ import division

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
from tqdm import tqdm
from os.path import join, dirname
import sys
import numpy as np
from point_viz.converter import PointvizConverter
HOME = join(dirname(os.getcwd()))
sys.path.append(HOME)

from models import rcnn_model as model
from train.configs import rcnn_config as config

data_home = '/home/tan/tony/dv-det/eval/data'
model_path = '/home/tan/tony/dv-det/checkpoints/stage2_van/test/best_model_0.7802847970985773'
# model_path = '/home/tan/tony/dv-det/checkpoints/stage2_l1/test/best_model_0.7438543078426844'
os.system("rm -r {}".format('/home/tan/tony/threejs/dv-det'))
Converter = PointvizConverter(home='/home/tan/tony/threejs/dv-det')
visualization = True

input_coors_stack = np.load(join(data_home, 'input_coors.npy'), allow_pickle=True)
input_features_stack = np.load(join(data_home, 'input_features.npy'), allow_pickle=True)
input_num_list_stack = np.load(join(data_home, 'input_num_list.npy'), allow_pickle=True)
input_bboxes_stack = np.load(join(data_home, 'input_bboxes.npy'), allow_pickle=True)

input_coors_p, input_features_p, input_num_list_p, input_bbox_p = model.stage1_inputs_placeholder()
is_training_p = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')


stage1_input_coors_p, stage1_input_features_p, stage1_input_num_list_p, _ = \
    model.stage1_inputs_placeholder(input_channels=1,
                                    bbox_padding=config.bbox_padding)

# stage2_input_coors_p, stage2_input_features_p, stage2_input_num_list_p, \
#     input_roi_coors_p, input_roi_attrs_p, input_roi_conf_logits_p, input_roi_num_list_p, input_bbox_p = \
#     model.stage2_inputs_placeholder()




is_stage1_training_p = tf.placeholder(dtype=tf.bool, shape=[], name="stage1_training")
is_stage2_training_p = tf.placeholder(dtype=tf.bool, shape=[], name="stage2_training")

stage1_output_coors, stage1_output_features, stage1_output_num_list, \
    output_roi_coors, output_roi_attrs, output_roi_conf_logits, output_roi_num_list = \
    model.stage1_model(input_coors=stage1_input_coors_p,
                       input_features=stage1_input_features_p,
                       input_num_list=stage1_input_num_list_p,
                       is_training=is_stage1_training_p,
                       is_eval=True,
                       trainable=False,
                       mem_saving=False,
                       bn=1.)

# roi_ious = model.get_roi_iou(roi_coors=output_roi_coors,
#                              pred_roi_attrs=output_roi_attrs,
#                              roi_num_list=output_roi_num_list,
#                              bbox_labels=input_bbox_p)

bbox_attrs, bbox_conf_logits, bbox_num_list, bbox_idx = \
    model.stage2_model(coors=stage1_output_coors,
                       features=stage1_output_features,
                       num_list=stage1_output_num_list,
                       roi_attrs=output_roi_attrs,
                       roi_conf_logits=output_roi_conf_logits,
                       roi_ious=output_roi_conf_logits,
                       roi_num_list=output_roi_num_list,
                       is_training=is_stage2_training_p,
                       trainable=False,
                       is_eval=True,
                       mem_saving=False,
                       bn=1.)
bbox_conf = tf.nn.sigmoid(bbox_conf_logits)

# bbox_attrs, bbox_conf, nms_idx, nms_count = \
#     rotated_nms3d(bbox_attrs, bbox_conf, nms_overlap_thresh=1e-3, nms_conf_thres=0.5)

stage1_loader = tf.train.Saver()
saver = tf.train.Saver(max_to_keep=None)

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()
tf_config = tf.ConfigProto()
tf_config.gpu_options.visible_device_list = "0"
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = False
tf_config.log_device_placement = False

if __name__ == '__main__':
    with tf.Session(config=tf_config) as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        saver.restore(sess, model_path)
        prediction_output = []
        for frame_id in tqdm(range(len(input_coors_stack))):
            batch_input_coors = input_coors_stack[frame_id]
            batch_input_features = input_features_stack[frame_id]
            batch_input_num_list = input_num_list_stack[frame_id]
            batch_input_bboxes = input_bboxes_stack[frame_id]

            # stage2_input_coors, stage2_input_features, stage2_input_num_list, \
            # input_roi_coors, input_roi_attrs, input_roi_conf_logits, input_roi_num_list = \
            #     sess.run([stage1_output_coors, stage1_output_features, stage1_output_num_list,
            #               output_roi_coors, output_roi_attrs, output_roi_conf_logits, output_roi_num_list],
            #              feed_dict={stage1_input_coors_p: batch_input_coors,
            #                         stage1_input_features_p: batch_input_features,
            #                         stage1_input_num_list_p: batch_input_num_list,
            #                         is_stage1_training_p: False})

            output_bboxes, output_coors, output_conf = \
                sess.run([bbox_attrs, stage1_output_coors, bbox_conf],
                         feed_dict={stage1_input_coors_p: batch_input_coors,
                                    stage1_input_features_p: batch_input_features,
                                    stage1_input_num_list_p: batch_input_num_list,
                                    is_stage1_training_p: False,
                                    is_stage2_training_p: False})
                         # options=run_options,
                         # run_metadata=run_metadata)
    #         if frame_id == 71:
    #             tl = timeline.Timeline(run_metadata.step_stats)
    #             ctf = tl.generate_chrome_trace_format()
    #             with open('timeline.json', 'w') as f:
    #                 f.write(ctf)
    #             break
    #
    #         output_idx = output_conf > 0.5
    #         # output_idx = output_idx[:output_count[0]]
    #         output_bboxes = output_bboxes[output_idx]
    #         output_conf = output_conf[output_idx]
    #         #
    #         input_rgbs = np.zeros_like(batch_input_coors) + [255, 255, 255]
    #         output_rgbs = np.zeros_like(output_coors) + [255, 0, 0]
    #         plot_coors = np.concatenate([batch_input_coors, output_coors], axis=0)
    #         plot_rgbs = np.concatenate([input_rgbs, output_rgbs], axis=0)
    #
    #         w = output_bboxes[:, 0]
    #         l = output_bboxes[:, 1]
    #         h = output_bboxes[:, 2]
    #         x = output_bboxes[:, 3]
    #         y = output_bboxes[:, 4]
    #         z = output_bboxes[:, 5]
    #         r = output_bboxes[:, 6]
    #
    #         c = np.zeros(len(w))
    #         d = np.zeros(len(w))
    #         pred_bboxes = np.stack([w, l, h, x, y, z, r, c, d], axis=-1)
    #         pred_bboxes = np.concatenate([pred_bboxes, np.expand_dims(output_conf, axis=-1)], axis=-1)
    #         prediction_output.append(pred_bboxes)
    #
    #         output_bboxes = input_bboxes_stack[frame_id][0]
    #         output_bboxes = output_bboxes[output_bboxes[:, 0] != 0, :]
    #         w = output_bboxes[:, 0]
    #         l = output_bboxes[:, 1]
    #         h = output_bboxes[:, 2]
    #         x = output_bboxes[:, 3]
    #         y = output_bboxes[:, 4]
    #         z = output_bboxes[:, 5]
    #         r = output_bboxes[:, 6]
    #         c = np.zeros(len(w))
    #         d = np.zeros(len(w))
    #         p = np.ones(len(w))
    #         label_bboxes = np.stack([w, l, h, x, y, z, r, c, d, p], axis=-1)
    #
    #
    #
    #         if visualization:
    #             # pred_bbox_params = convert_threejs_bbox_with_prob(pred_bboxes, color=output_conf) if len(pred_bboxes) > 0 else []
    #             pred_bbox_params = convert_threejs_bbox_with_colors(pred_bboxes, color='red') if len(pred_bboxes) > 0 else []
    #             label_bbox_params = convert_threejs_bbox_with_colors(label_bboxes, color='blue') if len(label_bboxes) > 0 else []
    #             task_name = "ID_%06d_%03d" % (frame_id, len(pred_bboxes))
    #
    #             Converter.compile(task_name=task_name,
    #                               coors=convert_threejs_coors(plot_coors),
    #                               default_rgb=plot_rgbs,
    #                               bbox_params=pred_bbox_params + label_bbox_params)
    # np.save(join(data_home, 'bbox_predictions.npy'), prediction_output)
