import os
from os.path import join
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.python.client import timeline
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

from data.kitti_generator import Dataset
import train.kitti.kitti_config as config
# tf.enable_eager_execution()
from models.tf_ops.loader.sampling import grid_sampling, get_bev_features
from models.tf_ops.loader.pooling import bev_projection
from models.tf_ops.loader.others import anchor_iou3d
from models.tf_ops.loader.bbox_utils import get_roi_bbox, get_anchor_attrs, get_bev_gt_bbox
from models.tf_ops.test.test_utils import fetch_instance, plot_points

from models.utils.iou_utils import cal_bev_iou


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 1
epoch = 2
dimension = [100., 140., 9.]
offset = [10., 70., 5.]
anchor_size = [1.6, 3.9, 1.5]

anchor_param_list = tf.constant([[1.6, 3.9, 1.5, -1.0, 0.],
                                 [1.6, 3.9, 1.5, -1.0, np.pi/2]])

if __name__ == '__main__':
    Dataset = Dataset(task='training',
                      batch_size=batch_size,
                      config=config.aug_config,
                      num_worker=6,
                      hvd_size=1,
                      hvd_id=0)
    coors, features, num_list, labels = next(Dataset.train_generator())
    Dataset.stop()

    coors = tf.cast(coors, dtype=tf.float32)
    coors, num_list, idx = grid_sampling(coors, num_list, 0.2, offset=offset, dimension=dimension)
    features = tf.gather(features, idx)

    bev_img = bev_projection(input_coors=coors,
                             input_features=features,
                             input_num_list=num_list,
                             resolution=0.2,
                             dimension=dimension,
                             offset=offset,
                             buffer_size=10)

    bev_coors, bev_features, bev_num_list = get_bev_features(bev_img=bev_img,
                                                             resolution=0.2,
                                                             offset=offset)

    # gt_roi_attrs, gt_roi_conf, gt_roi_diff = get_roi_bbox(input_coors=bev_coors,
    #                                                       bboxes=labels_p,
    #                                                       input_num_list=bev_num_list,
    #                                                       anchor_size=anchor_size,
    #                                                       expand_ratio=0.2,
    #                                                       diff_thres=config.diff_thres,
    #                                                       cls_thres=config.cls_thres)

    gt_attrs, gt_conf, gt_idx = get_bev_gt_bbox(input_coors=bev_coors,
                                                label_bbox=labels,
                                                input_num_list=bev_num_list,
                                                anchor_param_list=anchor_param_list,
                                                expand_ratio=0.15,
                                                diff_thres=4,
                                                cls_thres=1)

    anchor_attrs = get_anchor_attrs(anchor_coors=bev_coors,
                                    anchor_param_list=anchor_param_list)

    gt_attrs = tf.reshape(gt_attrs, [-1, tf.shape(gt_attrs)[2]])
    anchor_attrs = tf.reshape(anchor_attrs, [-1, tf.shape(anchor_attrs)[2]])
    bev_iou = cal_bev_iou(gt_attrs, anchor_attrs)



    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = False
    # config.allow_soft_placement = False
    # config.log_device_placement = False
    # config.gpu_options.visible_device_list = '0'
    # init_op = tf.initialize_all_variables()
    # with tf.Session(config=config) as sess:
    #     # sess.run(init_op)
    #     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #     run_metadata = tf.RunMetadata()
    #     for i in tqdm(range(epoch)):
    #         output_coors, output_num_list, output_img, output_attrs, output_ious = \
    #             sess.run([bev_coors, bev_num_list, bev_img, anchor_attrs, ious],
    #                      feed_dict={coors_p: input_coors[i],
    #                                 features_p: input_features[i],
    #                                 num_list_p: input_num_list[i],
    #                                 labels_p: input_labels[i]},
    #                      options=run_options,
    #                      run_metadata=run_metadata)
    #         tl = timeline.Timeline(run_metadata.step_stats)
    #         ctf = tl.generate_chrome_trace_format()
    #         with open('bev_proj_{}.json'.format(i), 'w') as f:
    #             f.write(ctf)
    #
    #         print("finished.")
    #
    input_coors = coors
    output_coors = np.concatenate([bev_coors.numpy(), np.zeros([bev_coors.numpy().shape[0], 1]) + -1.0], axis=-1)

    id = 0
    input_coors = fetch_instance(coors, num_list, id=id)
    input_rgbs = np.zeros_like(input_coors) + [255, 255, 255]


    output_coors = fetch_instance(output_coors, bev_num_list, id=id)
    output_rgbs = np.zeros_like(output_coors) + [255, 0, 0]

    plot_coors = np.concatenate([input_coors, output_coors], axis=0)
    plot_rgbs = np.concatenate([input_rgbs, output_rgbs], axis=0)

    output_attrs = fetch_instance(anchor_attrs, deepcopy(bev_num_list)*2, id=id).numpy()
    bev_iou = fetch_instance(bev_iou, deepcopy(bev_num_list)*2, id=id).numpy()
    output_attrs = output_attrs[bev_iou > 0.1, :]


    plot_points(coors=input_coors,
                rgb=input_rgbs,
                name='bev_coors',
                bboxes=output_attrs)

    # output_img = np.sum(output_idx >= 0, axis=-1)
