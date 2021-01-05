import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from tqdm import tqdm
import json
import io
import os
from point_viz.converter import PointvizConverter
Converter = PointvizConverter(home='/home/tan/tony/threejs/validation')

from data_utils.pc_generator import KITTI_PC as GENERATOR
from train.training_configs import bbox_reg_config as CONFIG
from models import rcnn_iou_paste as MODEL
from data_utils.normalization import convert_threejs_coors, convert_threejs_bbox_with_prob
from train_val_inference.nms import nms

# model_path = '/home/tan/tony/detection_reforge/checkpoints/test/test/best_model_0.8033773272984389'
# model_path = '/home/tan/tony/detection_reforge/checkpoints/rcnn_grid_2x_more/test/best_model_0.7462812192169908'
model_path = '/home/tan/tony/detection_reforge/checkpoints/rcnn_grid_sin/test/best_model_0.7770301869651364'
# model_path = '/media/data1/detection_reforge/checkpoints/test/test/best_model_0.765519233776142'

BATCH_SIZE = 64
PHASE = 'training'
anchor_size = tf.constant([1.6, 3.9, 1.5])

Dataset = GENERATOR(phase=PHASE,
                    batch_size=BATCH_SIZE,
                    validation=True,
                    # use_trimmed_foreground=CONFIG.use_trimmed_foreground,
                    use_trimmed_foreground=False,
                    normalization=CONFIG.normalization)

input_coors_stack = []
input_features_stack = []
input_num_list_stack = []
input_bbox_stack = []
original_coors_stack = np.load('/home/tan/tony/detection_reforge/dataset/lidar_points_{}.npy'.format(PHASE), allow_pickle=True)[..., :3]

print("INFO: Loading dataset...")
for i in tqdm(range(Dataset.batch_sum)):
    batch_input_coors, batch_input_features, batch_input_num_list, batch_input_bbox = next(Dataset.valid_generator())
    input_coors_stack.append(batch_input_coors)
    input_features_stack.append(batch_input_features)
    input_num_list_stack.append(batch_input_num_list)
    input_bbox_stack.append(batch_input_bbox)
print("INFO: Loading completed.")


input_coors_p, input_features_p, input_num_list_p, input_bbox_p = \
    MODEL.stage1_inputs_placeholder(input_channels=1,
                                    bbox_padding=CONFIG.bbox_padding)
is_roi_training_p = tf.placeholder(dtype=tf.bool, shape=[], name="roi_training")
is_bbox_training_p = tf.placeholder(dtype=tf.bool, shape=[], name="bbox_training")

roi_base_coors, roi_attrs, bbox_attrs, roi_cls_logits, bbox_cls_logits, bbox_num_list = \
    MODEL.model(coors=input_coors_p,
                features=input_features_p,
                num_list=input_num_list_p,
                is_roi_training=is_roi_training_p,
                is_bbox_training=is_bbox_training_p,
                config=CONFIG,
                roi_bn=1.,
                bbox_bn=1.)

roi_conf = tf.nn.sigmoid(roi_cls_logits)
bbox_conf = tf.nn.sigmoid(bbox_cls_logits)


saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0"
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False

frame_id = 0
final_output_bboxes = []

if __name__ == '__main__':
    with tf.Session(config=config) as sess:
        saver.restore(sess, model_path)
        print("INFO: Inference begin.")
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        for b in tqdm(range(Dataset.batch_sum)):
            batch_input_coors = input_coors_stack[b]
            batch_input_features = input_features_stack[b]
            batch_input_num_list = input_num_list_stack[b]
            batch_input_bbox = input_bbox_stack[b]
            batch_output_bbox_attrs, batch_output_bbox_conf, batch_output_coors = \
                sess.run([roi_attrs, roi_conf, roi_base_coors],
                         feed_dict={input_coors_p: batch_input_coors,
                                    input_features_p: batch_input_features,
                                    input_num_list_p: batch_input_num_list,
                                    input_bbox_p: batch_input_bbox,
                                    is_roi_training_p: False,
                                    is_bbox_training_p: False})
            # if b > 30:
            #     tl = timeline.Timeline(run_metadata.step_stats)
            #     ctf = tl.generate_chrome_trace_format()
            #
            #     with open('timeline.json', 'w') as f:
            #         f.write(ctf)
            #     break


    #         batch_size = len(batch_input_num_list)
    #         input_start_idx = 0
    #         output_start_idx = 0
    #         for i in range(batch_size):
    #             original_coors = original_coors_stack[frame_id]
    #             original_rgbs = np.zeros_like(original_coors) + [255, 255, 255]  # [n, 3]
    #             bbox_labels = batch_input_bbox[i]
    #
    #             input_stop_idx = input_start_idx + batch_input_num_list[i]
    #             input_coors = batch_input_coors[input_start_idx:input_stop_idx, :]
    #
    #             output_stop_idx = output_start_idx + batch_output_num_list[i]
    #             output_coors = batch_output_coors[output_start_idx:output_stop_idx, :]
    #             output_bbox_attrs = batch_output_bbox_attrs[output_start_idx:output_stop_idx, :]
    #             output_conf = batch_output_bbox_conf[output_start_idx:output_stop_idx]
    #
    #
    #             plot_coors = original_coors
    #             plot_rgbs = original_rgbs
    #
    #             if len(input_coors) > 0:
    #                 input_rgbs = np.zeros_like(input_coors)
    #                 plot_coors = np.concatenate([plot_coors, input_coors], axis=0)
    #                 plot_rgbs = np.concatenate([plot_rgbs, input_rgbs], axis=0)
    #             if len(output_coors) > 0:
    #                 output_rgbs = np.zeros_like(output_coors) + [255, 0, 0]
    #                 plot_coors = np.concatenate([plot_coors, output_coors], axis=0)
    #                 plot_rgbs = np.concatenate([plot_rgbs, output_rgbs], axis=0)
    #
    #
    #
    #
    #             mask = output_conf > 0.5
    #             output_conf = output_conf[mask]
    #             output_bboxes = output_bbox_attrs[mask, :]
    #             w = np.min(output_bboxes[:, :2], axis=-1)
    #             l = np.max(output_bboxes[:, :2], axis=-1)
    #             h = output_bboxes[:, 2]
    #             x = output_bboxes[:, 3]
    #             y = output_bboxes[:, 4]
    #             z = output_bboxes[:, 5]
    #             r = output_bboxes[:, 6] + np.greater(output_bboxes[:, 0], output_bboxes[:, 1]).astype(np.float32) * np.pi / 2
    #             c = np.zeros(len(w))
    #             d = np.zeros(len(w))
    #             pred_bboxes = np.stack([w, l, h, x, y, z, r, c, d], axis=-1)
    #             pred_bboxes = np.concatenate([pred_bboxes, np.expand_dims(output_conf, axis=-1)], axis=-1)
    #             nms_bboxes, bbox_collection = nms(pred_bboxes, thres=1e-3)
    #             # bboxes = nms_average(bbox_collection)
    #             final_output_bboxes.append(nms_bboxes)
    #
    #             if len(nms_bboxes) == 0:
    #                 print("WARNING: frame {} got empty prediction.".format(frame_id))
    #
    #             pred_bbox_params = convert_threejs_bbox_with_prob(nms_bboxes, color=output_conf) if len(nms_bboxes) > 0 else []
    #             gt_bbox_params = convert_threejs_bbox_with_prob(bbox_labels)
    #             bbox_params =  pred_bbox_params + gt_bbox_params
    #             task_name = "ID_%06d_%02d-%02d" % (frame_id , len(gt_bbox_params), len(pred_bbox_params))
    #
    #             # Converter.compile(task_name=task_name,
    #             #                   coors=convert_threejs_coors(plot_coors),
    #             #                   default_rgb=plot_rgbs,
    #             #                   bbox_params=bbox_params)
    #
    #             input_start_idx = input_stop_idx
    #             output_start_idx = output_stop_idx
    #             frame_id += 1
    #
    # np.save('/home/tan/tony/detection_reforge/eval/bbox_validation.npy', final_output_bboxes)
