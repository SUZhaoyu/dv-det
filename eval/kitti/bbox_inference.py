import os
from os.path import join

import numpy as np
import tensorflow as tf
from point_viz.converter import PointvizConverter

from tqdm import tqdm
import horovod.tensorflow as hvd

os.system("rm -r {}".format('/home/tan/tony/threejs/kitti-stage2'))
Converter = PointvizConverter(home='/home/tan/tony/threejs/kitti-stage2')

from models import kitti_model as model
from models.tf_ops.loader.others import rotated_nms3d_idx
from data.utils.normalization import convert_threejs_bbox_with_colors, convert_threejs_coors
from train.kitti import kitti_config as config

hvd.init()

# model_path = '/home/tan/tony/dv-det/checkpoints/stage1/test/best_model_0.6461553027390907'
# model_path = '/home/tan/tony/dv-det/checkpoints/stage2_heavy/test/best_model_0.7809948543101326'
# model_path = '/home/tan/tony/dv-det/ckpt-kitti/stage2/test/best_model_0.7913361259170595'
# model_path = '/home/tan/tony/dv-det/ckpt-kitti/stage2-test-2/test/best_model_0.7946715183855744'
# model_path = '/home/tan/tony/dv-det/ckpt-kitti/stage2/test/best_model_0.7960067724776407'
# model_path = '/home/tan/tony/dv-det/ckpt-kitti/stage2-conf=0.75/test/best_model_0.8025257015611262'
model_path = '/home/tan/tony/dv-det/ckpt-kitti/test/test/model_0.8062656926309775'
data_home = '/home/tan/tony/dv-det/eval/kitti/data'
# ******** Easy: 91.15, Moderate: 81.23, Hard: 74.94 ********
visualization = True

input_coors_stack = np.load(join(data_home, 'input_coors.npy'), allow_pickle=True)
input_features_stack = np.load(join(data_home, 'input_features.npy'), allow_pickle=True)
input_num_list_stack = np.load(join(data_home, 'input_num_list.npy'), allow_pickle=True)
input_bboxes_stack = np.load(join(data_home, 'input_bboxes.npy'), allow_pickle=True)

input_coors_p, input_features_p, input_num_list_p, _ = model.stage1_inputs_placeholder(
    input_channels=1,
    bbox_padding=config.bbox_padding)
is_training_p = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')


coors, features, num_list, roi_coors, roi_attrs, roi_conf_logits, roi_num_list = \
    model.stage1_model(input_coors=input_coors_p,
                       input_features=input_features_p,
                       input_num_list=input_num_list_p,
                       is_training=is_training_p,
                       is_eval=True,
                       trainable=False,
                       mem_saving=False,
                       bn=1.)
roi_conf = tf.nn.sigmoid(roi_conf_logits)

# nms_idx = rotated_nms3d_idx(roi_attrs, roi_conf, nms_overlap_thresh=0.8, nms_conf_thres=0.75)
# roi_attrs = tf.gather(roi_attrs, nms_idx, axis=0)
# roi_conf_logits = tf.gather(roi_conf_logits, nms_idx, axis=0)
# roi_num_list = tf.expand_dims(tf.shape(nms_idx)[0], axis=0)

bbox_attrs, bbox_conf_logits, bbox_dir_logits, bbox_num_list, bbox_idx = \
    model.stage2_model(coors=coors,
                       features=features,
                       num_list=num_list,
                       roi_attrs=roi_attrs,
                       roi_conf_logits=roi_conf_logits,
                       roi_ious=roi_conf_logits,
                       roi_num_list=roi_num_list,
                       is_training=is_training_p,
                       trainable=False,
                       is_eval=True,
                       mem_saving=False,
                       bn=1.)

bbox_conf = tf.nn.sigmoid(bbox_conf_logits)
bbox_dir = tf.nn.sigmoid(bbox_dir_logits)

nms_idx = rotated_nms3d_idx(bbox_attrs, bbox_conf, nms_overlap_thresh=0.1, nms_conf_thres=0.3)

loader = tf.train.Saver()
tf_config = tf.ConfigProto()
tf_config.gpu_options.visible_device_list = "0"
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = False
tf_config.log_device_placement = False


if __name__ == '__main__':
    with tf.Session(config=tf_config) as sess:
        loader.restore(sess, model_path)
        prediction_output = []
        for frame_id in tqdm(range(len(input_coors_stack))):
            batch_input_coors = input_coors_stack[frame_id]
            batch_input_features = input_features_stack[frame_id]
            batch_input_num_list = input_num_list_stack[frame_id]
            # batch_input_bboxes = input_bboxes_stack[frame_id]
            output_bboxes, output_coors, output_conf, output_dir, output_idx = \
                sess.run([bbox_attrs, coors, bbox_conf, bbox_dir, nms_idx],
            # output_bboxes, output_coors, output_conf = \
            #     sess.run([bbox_attrs, coors, bbox_conf],
                         feed_dict={input_coors_p: batch_input_coors,
                                    input_features_p: batch_input_features,
                                    input_num_list_p: batch_input_num_list,
                                    is_training_p: False})

            # output_idx = output_conf > 0.4
            # output_idx = output_idx[:output_count[0]]
            output_bboxes = output_bboxes[output_idx]
            output_conf = output_conf[output_idx]
            output_dir = np.array(output_dir[output_idx] > 0.5, dtype=np.float32)
            #
            input_rgbs = np.zeros_like(batch_input_coors) + [255, 255, 255]
            output_rgbs = np.zeros_like(output_coors) + [255, 0, 0]
            plot_coors = np.concatenate([batch_input_coors, output_coors], axis=0)
            plot_rgbs = np.concatenate([input_rgbs, output_rgbs], axis=0)

            w = output_bboxes[:, 0]
            l = output_bboxes[:, 1]
            h = output_bboxes[:, 2]
            x = output_bboxes[:, 3]
            y = output_bboxes[:, 4]
            z = output_bboxes[:, 5]
            r = output_bboxes[:, 6] + np.pi * output_dir

            c = np.zeros(len(w))
            d = np.zeros(len(w))
            pred_bboxes = np.stack([w, l, h, x, y, z, r, c, d], axis=-1)
            pred_bboxes = np.concatenate([pred_bboxes, np.expand_dims(output_conf, axis=-1)], axis=-1)
            prediction_output.append(pred_bboxes)

            output_bboxes = input_bboxes_stack[frame_id][0]
            output_bboxes = output_bboxes[output_bboxes[:, 0] != 0, :]
            w = output_bboxes[:, 0]
            l = output_bboxes[:, 1]
            h = output_bboxes[:, 2]
            x = output_bboxes[:, 3]
            y = output_bboxes[:, 4]
            z = output_bboxes[:, 5]
            r = output_bboxes[:, 6]
            c = np.zeros(len(w))
            d = output_bboxes[:, 8]
            p = np.ones(len(w))
            label_bboxes = np.stack([w, l, h, x, y, z, r, c, d, p], axis=-1)


            if visualization:
                # pred_bbox_params = convert_threejs_bbox_with_prob(pred_bboxes, color=output_conf) if len(pred_bboxes) > 0 else []
                pred_bbox_params = convert_threejs_bbox_with_colors(pred_bboxes, color='red') if len(pred_bboxes) > 0 else []
                label_bbox_params = convert_threejs_bbox_with_colors(label_bboxes, color='blue') if len(label_bboxes) > 0 else []
                task_name = "ID_%06d_%03d" % (frame_id, len(pred_bboxes))

                Converter.compile(task_name=task_name,
                                  coors=convert_threejs_coors(plot_coors),
                                  default_rgb=plot_rgbs,
                                  bbox_params=pred_bbox_params + label_bbox_params)
    np.save(join(data_home, 'bbox_predictions.npy'), prediction_output)
