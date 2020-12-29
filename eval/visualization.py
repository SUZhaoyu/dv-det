import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from tqdm import tqdm
import json
import io
import os
from os.path import join
from point_viz.converter import PointvizConverter
Converter = PointvizConverter(home='/home/tan/tony/threejs/dv-det')

from train.configs import rcnn_config as config
from models import rcnn_single_stage as model
from data.utils.normalization import convert_threejs_coors, convert_threejs_bbox_with_prob

model_path = '/home/tan/tony/dv-det/checkpoints/test_dense/test/best_model_0.6230139189799166'
data_home = '/home/tan/tony/dv-det/eval/data'

input_coors_stack = np.load(join(data_home, 'input_coors.npy'), allow_pickle=True)
input_features_stack = np.load(join(data_home, 'input_features.npy'), allow_pickle=True)
input_num_list_stack = np.load(join(data_home, 'input_num_list.npy'), allow_pickle=True)

input_coors_p, input_features_p, input_num_list_p, _ = model.inputs_placeholder()
is_training_p = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

base_coors, roi_attrs, roi_conf_logits, num_list = \
    model.model(input_coors=input_coors_p,
                input_features=input_features_p,
                input_num_list=input_num_list_p,
                is_training=is_training_p,
                config=config,
                bn=1.)
roi_conf = tf.nn.sigmoid(roi_conf_logits)
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0"
config.gpu_options.allow_growth = False
config.allow_soft_placement = False
config.log_device_placement = False


if __name__ == '__main__':
    with tf.Session(config=config) as sess:
        saver.restore(sess, model_path)
        for frame_id in tqdm(range(len(input_coors_stack))):
            batch_input_coors = input_coors_stack[frame_id]
            batch_input_features = input_features_stack[frame_id]
            batch_input_num_list = input_num_list_stack[frame_id]
            output_attrs, output_conf, output_coors = \
                sess.run([roi_attrs, roi_conf, base_coors],
                         feed_dict={input_coors_p: batch_input_coors,
                                    input_features_p: batch_input_features,
                                    input_num_list_p: batch_input_num_list,
                                    is_training_p: False})

            input_rgbs = np.zeros_like(batch_input_coors) + [255, 255, 255]
            output_rgbs = np.zeros_like(output_coors) + [255, 0, 0]
            plot_coors = np.concatenate([batch_input_coors, output_coors], axis=0)
            plot_rgbs = np.concatenate([input_rgbs, output_rgbs], axis=0)

            mask = output_conf > 0.5
            output_conf = output_conf[mask]
            output_bboxes = output_attrs[mask, :]
            w = np.min(output_bboxes[:, :2], axis=-1)
            l = np.max(output_bboxes[:, :2], axis=-1)
            h = output_bboxes[:, 2]
            x = output_bboxes[:, 3]
            y = output_bboxes[:, 4]
            z = output_bboxes[:, 5]
            r = output_bboxes[:, 6] + np.greater(output_bboxes[:, 0], output_bboxes[:, 1]).astype(
                np.float32) * np.pi / 2
            c = np.zeros(len(w))
            d = np.zeros(len(w))
            pred_bboxes = np.stack([w, l, h, x, y, z, r, c, d], axis=-1)
            pred_bboxes = np.concatenate([pred_bboxes, np.expand_dims(output_conf, axis=-1)], axis=-1)


            pred_bbox_params = convert_threejs_bbox_with_prob(pred_bboxes, color=output_conf) if len(
                pred_bboxes) > 0 else []
            task_name = "ID_%06d_%03d" % (frame_id, len(pred_bboxes))

            Converter.compile(task_name=task_name,
                              coors=convert_threejs_coors(plot_coors),
                              default_rgb=plot_rgbs,
                              bbox_params=pred_bbox_params)
