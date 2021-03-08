import os

import tensorflow as tf
from tqdm import tqdm
from tensorflow.python.client import timeline
from data.kitti_generator import Dataset
# tf.enable_eager_execution()
from models.tf_ops.loader.sampling import grid_sampling_thrust, voxel_sampling_feature, voxel_sampling_idx, voxel_sampling_idx_binary
from models.tf_ops.loader.bbox_utils import get_roi_bbox
import train.kitti.kitti_config as config
from data.utils.normalization import convert_threejs_coors, convert_threejs_bbox, convert_threejs_bbox_with_assigned_colors
from point_viz.converter import PointvizConverter
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colors
# print(colors.to_rgba('blue'))
CSS_COLOR_LIST = list(mcolors.CSS4_COLORS)
Converter = PointvizConverter(home='/home/tan/tony/threejs')
import numpy as np
# from models.utils.ops_wrapper import kernel_conv_wrapper
from models.tf_ops.test.test_utils import fetch_instance, get_rgbs_from_coors, plot_points_from_voxels_with_color, \
    get_rgbs_from_coors_tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


batch_size = 4
epoch = 1
if __name__ == '__main__':
    KittiDataset = Dataset(task='training',
                           batch_size=batch_size,
                           config=config.aug_config,
                           num_worker=6,
                           hvd_size=1,
                           hvd_id=0,
                           validation=False)
    input_coors, _, input_num_list, input_bbox = next(KittiDataset.train_generator())

    KittiDataset.stop()
    # np.save("input_coors.npy", input_coors)
    # np.save("input_features.npy", input_features)
    # np.save("input_num_list.npy", input_num_list)

    # input_coors = np.load("input_coors.npy", allow_pickle=True)
    # input_features = np.load("input_features.npy", allow_pickle=True)
    # input_num_list = np.load("input_num_list.npy", allow_pickle=True)

    coors_p = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    features_p = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    num_list_p = tf.placeholder(dtype=tf.int32, shape=[None])
    bbox_labels_p = tf.placeholder(dtype=tf.float32, shape=[None, 128, 9])
    coors, num_list = coors_p, num_list_p

    # coors, features, num_list, voxels = point_sampling(coors, features, num_list, 16,0.8, 'layer_0')
    coors_0, num_list_0, _ = grid_sampling_thrust(coors_p, num_list_p, 0.1, dimension=config.dimension_training, offset=config.offset_training)
    coors_1, num_list_1, _ = grid_sampling_thrust(coors_0, num_list_0, 0.2, dimension=config.dimension_training, offset=config.offset_training)
    coors_2, num_list_2, _ = grid_sampling_thrust(coors_1, num_list_1, 0.6, dimension=config.dimension_training, offset=config.offset_training)
    coors_3, num_list_3, _ = grid_sampling_thrust(coors_2, num_list_2, 1.4, dimension=config.dimension_training, offset=config.offset_training)

    gt_roi_attrs, gt_roi_conf, gt_roi_diff = get_roi_bbox(input_coors=coors_3,
                                                          bboxes=bbox_labels_p,
                                                          input_num_list=num_list_3,
                                                          anchor_size=config.anchor_size,
                                                          expand_ratio=0.2,
                                                          diff_thres=config.diff_thres,
                                                          cls_thres=config.cls_thres)

    init_op = tf.initialize_all_variables()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.allow_soft_placement = False
    config.log_device_placement = False
    config.gpu_options.visible_device_list = '0'
    with tf.Session(config=config) as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        for i in tqdm(range(epoch)):
            output_coors, output_num_list, output_bboxes, output_conf = sess.run([coors_3, num_list_3, gt_roi_attrs, gt_roi_conf],
                                                                        feed_dict={coors_p: input_coors,
                                                                                   num_list_p: input_num_list,
                                                                                   bbox_labels_p: input_bbox})

    id = 2

    input_coors = fetch_instance(input_coors, input_num_list, id=id)
    output_coors = fetch_instance(output_coors, output_num_list, id=id)
    output_bboxes = fetch_instance(output_bboxes, output_num_list, id=id)
    output_conf = fetch_instance(output_conf, output_num_list, id=id)

    output_idx = output_conf > 0
    output_bboxes = output_bboxes[output_idx]
    output_conf = output_conf[output_idx]
    output_coors = output_coors[output_idx]

    w = output_bboxes[:, 0] * ((np.random.rand(len(output_bboxes)) * 0.1 - 0.05) + 1.)
    l = output_bboxes[:, 1] * ((np.random.rand(len(output_bboxes)) * 0.1 - 0.05) + 1.)
    h = output_bboxes[:, 2] * ((np.random.rand(len(output_bboxes)) * 0.1 - 0.05) + 1.)
    x = output_bboxes[:, 3] + (np.random.rand(len(output_bboxes)) * 0.2 - 0.1)
    y = output_bboxes[:, 4] + (np.random.rand(len(output_bboxes)) * 0.2 - 0.1)
    z = output_bboxes[:, 5] + (np.random.rand(len(output_bboxes)) * 0.2 - 0.1)
    r = output_bboxes[:, 6] * ((np.random.rand(len(output_bboxes)) * 0.2 - 0.1) + 1.)

    c = np.zeros(len(w))
    d = np.zeros(len(w))
    gt_bboxes = np.stack([w, l, h, x, y, z, r, c, d], axis=-1)
    gt_bboxes = np.concatenate([gt_bboxes, np.expand_dims(output_conf, axis=-1)], axis=-1)

    output_color_rgbs = []
    output_color_names = []
    for i in range(len(output_coors)):
        color_name = random.choice(CSS_COLOR_LIST)
        output_color_rgbs.append(np.array(colors.to_rgb(color_name))*255)
        output_color_names.append(color_name)

    input_rgbs = np.zeros_like(input_coors) + [255, 255, 255]
    plot_coors = np.concatenate([input_coors, output_coors], axis=0)
    plot_rgbs = np.concatenate([input_rgbs, output_color_rgbs], axis=0)

    label_bbox_params = convert_threejs_bbox_with_assigned_colors(gt_bboxes, colors=output_color_names) if len(gt_bboxes) > 0 else []

    Converter.compile(task_name='gt_bboxes_aug',
                      coors=convert_threejs_coors(plot_coors),
                      default_rgb=plot_rgbs,
                      bbox_params=label_bbox_params)
