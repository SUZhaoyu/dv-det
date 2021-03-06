import os
from os.path import join
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import time
import tensorflow as tf
from point_viz.converter import PointvizConverter
from tensorflow.python.client import timeline
from tqdm import tqdm
# from models.tf_ops.loader.others import rotated_nms3d_idx
from data.utils.normalization import convert_threejs_coors, convert_threejs_bbox_with_colors
import horovod.tensorflow as hvd

hvd.init()

os.system("rm -r {}".format('/home/tan/tony/threejs/waymo-stage1'))
Converter = PointvizConverter(home='/home/tan/tony/threejs/waymo-stage1')

from models import waymo_model as model

# model_path = '/home/tan/tony/dv-det/checkpoints/stage1_eval/test/best_model_0.6925084921062944' # 68.8%@non-mem-saving
# model_path = '/home/tan/tony/dv-det/checkpoints/stage1_van/test/best_model_0.672630966259817' # 68.8%@non-mem-saving
# model_path = '/home/tan/tony/dv-det/checkpoints/waymo-stage1-no_scale_lr/test/best_model_0.6559794634147377' # 68.8%@non-mem-saving
model_path = '/home/tan/tony/dv-det/ckpt-waymo/stage1-complicated-paste/test/best_model_0.6200861480780779' # 68.8%@non-mem-saving
# model_path = '/home/tan/tony/dv-det/checkpoints/waymo-stage1-avg_pool/test/best_model_0.6606513755109455' # 68.8%@non-mem-saving
# model_path = '/home/tan/tony/dv-det/checkpoints/stage1_focal=0.75/test/best_model_0.6906541874676403' # 68.5%@non-mem-saving
data_home = '/home/tan/tony/dv-det/eval/waymo/data'
visualization = True

# DatasetTrain = Dataset(task="train",
#                        batch_size=16,
#                        config=config.aug_config,
#                        num_worker=config.num_worker,
#                        hvd_size=hvd.size(),
#                        hvd_id=hvd.rank())
# dataset_generator = DatasetTrain.train_generator()

input_coors_stack = np.load(join(data_home, 'input_coors.npy'), allow_pickle=True)
input_features_stack = np.load(join(data_home, 'input_features.npy'), allow_pickle=True)
input_num_list_stack = np.load(join(data_home, 'input_num_list.npy'), allow_pickle=True)
input_bboxes_stack = np.load(join(data_home, 'input_bboxes.npy'), allow_pickle=True)

# batch_input_coors = np.load(join(data_home, 'input_coors_nan.npy'), allow_pickle=True)
# batch_input_features = np.load(join(data_home, 'input_features_nan.npy'), allow_pickle=True)
# batch_input_num_list = np.load(join(data_home, 'input_num_list_nan.npy'), allow_pickle=True)
# batch_input_bboxes = np.load(join(data_home, 'input_bboxes_nan.npy'), allow_pickle=True)


input_coors_p, input_features_p, input_num_list_p, input_bbox_p = model.stage1_inputs_placeholder(input_channels=2)
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

# nms_idx = rotated_nms3d_idx(roi_attrs, roi_conf, nms_overlap_thresh=0.75, nms_conf_thres=0.6)

# gt_roi_attrs, gt_roi_conf, gt_roi_diff = get_roi_bbox(input_coors=roi_coors,
#                                                       bboxes=input_bbox_p,
#                                                       input_num_list=roi_num_list,
#                                                       anchor_size=config.anchor_size,
#                                                       expand_ratio=0.2,
#                                                       diff_thres=4)
# roi_ious = cal_3d_iou(gt_attrs=gt_roi_attrs, pred_attrs=roi_attrs, clip=False)
# roi_iou_masks = tf.cast(tf.equal(gt_roi_conf, 1), dtype=tf.float32) # [-1, 0, 1] -> [0, 0, 1]
# roi_iou_loss = get_masked_average(1. - roi_ious, roi_iou_masks)


# roi_attrs, roi_conf, roi_coors, nms_idx, nms_count = \
#     rotated_nms3d(roi_attrs, roi_conf, roi_coors, nms_overlap_thresh=0.7, nms_conf_thres=0.4)
#
# gt_roi_attrs, gt_roi_conf, gt_roi_diff = get_roi_bbox(input_coors=roi_coors,
#                                                       bboxes=input_bbox_p,
#                                                       input_num_list=roi_num_list,
#                                                       anchor_size=config.anchor_size,
#                                                       expand_ratio=0.2,
#                                                       diff_thres=4)
# #
# roi_ious = cal_3d_iou(gt_attrs=gt_roi_attrs, pred_attrs=roi_attrs, clip=False)

# init_op = tf.initialize_all_variables()
stage1_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stage1')
saver = tf.train.Saver(stage1_variables)
config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0"
config.gpu_options.allow_growth = True
config.allow_soft_placement = False
config.log_device_placement = False


if __name__ == '__main__':
    with tf.Session(config=config) as sess:
        saver.restore(sess, model_path)
        prediction_output = []
        label_output = []
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        for frame_id in tqdm(range(len(input_coors_stack))):
        # for _ in tqdm(range(100000)):
            # batch_input_coors, batch_input_features, batch_input_num_list, batch_input_bboxes = \
            #     next(dataset_generator)
            batch_input_coors = input_coors_stack[frame_id]
            batch_input_features = input_features_stack[frame_id]
            batch_input_num_list = input_num_list_stack[frame_id]
            batch_input_bboxes = input_bboxes_stack[frame_id]
            # output_bboxes, output_coors, output_conf, output_ious, output_diff, output_idx, output_count = \
            #     sess.run([roi_attrs, roi_coors, roi_conf, roi_ious, gt_roi_diff, nms_idx, nms_count],
            output_bboxes, output_coors, output_features, output_conf = \
                sess.run([roi_attrs, coors, features, roi_conf],
                         feed_dict={input_coors_p: batch_input_coors,
                                    input_features_p: batch_input_features,
                                    input_num_list_p: batch_input_num_list,
                                    input_bbox_p: batch_input_bboxes,
                                    is_training_p: False})
                        # options=run_options,
                        # run_metadata=run_metadata)

            time.sleep(0.5)

            # if output_iou_loss == np.nan:
            # print(output_iou_loss)
            # if np.isnan(np.mean(output_ious)):
            #     # print(np.average(output_ious), np.average(output_intersections))
            #     np.save(join(data_home, 'gt_roi_attrs_nan.npy'), output_gt_bboxes)
            #     np.save(join(data_home, 'pred_roi_attrs_nan.npy'), output_bboxes)
            #     np.save(join(data_home, 'roi_ious_nan.npy'), output_ious)
            #     break



            # tl = timeline.Timeline(run_metadata.step_stats)
            # ctf = tl.generate_chrome_trace_format()
            # with open('waymo-stage1.json', 'w') as f:
            #     f.write(ctf)

        #     output_idx = output_conf >= 0.6
        #     output_bboxes = output_bboxes[output_idx]
        #     output_conf = output_conf[output_idx]
        #
        #     # output_ious = output_ious[output_idx]
        #     # for i in range(len(output_ious)):
        #     #     if not np.isnan(output_ious[i]):
        #     #         overall_iou.append(output_ious[i])
        #     # #
        #     input_rgbs = np.zeros_like(batch_input_coors) + [255, 255, 255]
        #     output_rgbs = np.zeros_like(output_coors) + [255, 0, 0]
        #     plot_coors = np.concatenate([batch_input_coors, output_coors], axis=0)
        #     plot_rgbs = np.concatenate([input_rgbs, output_rgbs], axis=0)
        #
        #     w = output_bboxes[:, 0]
        #     l = output_bboxes[:, 1]
        #     h = output_bboxes[:, 2]
        #     x = output_bboxes[:, 3]
        #     y = output_bboxes[:, 4]
        #     z = output_bboxes[:, 5]
        #     r = output_bboxes[:, 6]
        #
        #     c = np.ones(len(w))
        #     d = np.zeros(len(w))
        #     pred_bboxes = np.stack([w, l, h, x, y, z, r, c, d], axis=-1)
        #     pred_bboxes = np.concatenate([pred_bboxes, np.expand_dims(output_conf, axis=-1)], axis=-1)
        #     prediction_output.append(pred_bboxes)
        #
        #     output_bboxes = input_bboxes_stack[frame_id][0]
        #     output_bboxes = output_bboxes[output_bboxes[:, 0] != 0, :]
        #     w = output_bboxes[:, 0]
        #     l = output_bboxes[:, 1]
        #     h = output_bboxes[:, 2]
        #     x = output_bboxes[:, 3]
        #     y = output_bboxes[:, 4]
        #     z = output_bboxes[:, 5]
        #     r = output_bboxes[:, 6]
        #     c = np.ones(len(w))
        #     d = output_bboxes[:, 8]
        #     p = np.ones(len(w))
        #     label_bboxes = np.stack([w, l, h, x, y, z, r, c, d, p], axis=-1)
        #     label_output.append(label_bboxes)
        #
        #
        #
        #     if visualization:
        #         # pred_bbox_params = convert_threejs_bbox_with_prob(pred_bboxes, color=output_conf) if len(pred_bboxes) > 0 else []
        #         pred_bbox_params = convert_threejs_bbox_with_colors(pred_bboxes, color='red') if len(pred_bboxes) > 0 else []
        #         label_bbox_params = convert_threejs_bbox_with_colors(label_bboxes, color='blue') if len(label_bboxes) > 0 else []
        #         task_name = "ID_%06d_%03d" % (frame_id, len(pred_bboxes))
        #
        #         Converter.compile(task_name=task_name,
        #                           coors=convert_threejs_coors(plot_coors),
        #                           default_rgb=plot_rgbs,
        #                           bbox_params=pred_bbox_params + label_bbox_params)
        #
        # np.save(join(data_home, 'bbox_predictions.npy'), np.array(prediction_output, dtype=object))
        # np.save(join(data_home, 'bbox_labels.npy'), np.array(label_output, dtype=object))