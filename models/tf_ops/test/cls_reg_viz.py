import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
from models.tf_ops.loader.others import anchor_iou3d, anchor_iou_filter
from models.tf_ops.loader.bbox_utils import get_roi_bbox, get_anchor_attrs, get_bev_gt_bbox
from models.tf_ops.test.test_utils import fetch_instance, plot_points
from models.utils.loss_utils import get_bbox_target_params

from models.utils.iou_utils import cal_bev_iou


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 6
epoch = 2
dimension = [100., 140., 9.]
offset = [10., 70., 5.]
anchor_size = [1.6, 3.9, 1.5]
delta_bin_xy = 0.5
scope_xy = 3.
delta_bin_angle = np.pi / 6.

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
    output_coors, output_num_list, idx = grid_sampling(coors, num_list, 0.4, offset=offset, dimension=dimension)
    features = tf.gather(features, idx)

    gt_roi_attrs, gt_roi_conf, gt_roi_diff = get_roi_bbox(input_coors=output_coors,
                                                          bboxes=labels,
                                                          input_num_list=output_num_list,
                                                          anchor_size=anchor_size,
                                                          expand_ratio=0.2,
                                                          diff_thres=config.diff_thres,
                                                          cls_thres=config.cls_thres)

    target_cls, target_res = get_bbox_target_params(point_coors=output_coors,
                                                    label_attrs=gt_roi_attrs,
                                                    anchor_size=anchor_size)

    bbox_offset_xy = -scope_xy + tf.cast(target_cls[:, :2], dtype=tf.float32) * delta_bin_xy + target_res[:, 3:5] * delta_bin_xy
    bbox_offset_z = target_res[:, 5:6]
    bbox_coors = output_coors + tf.concat([bbox_offset_xy, bbox_offset_z], axis=-1)

    bbox_dimension = tf.exp(target_res[:, :3]) * anchor_size
    bbox_r = -np.pi + tf.cast(target_cls[:, 2:3], dtype=tf.float32) * delta_bin_angle + target_res[:, 6:7] * delta_bin_angle

    target_attrs = tf.concat([bbox_dimension, bbox_coors, bbox_r], axis=-1)

    id = 3
    input_coors = fetch_instance(coors, num_list, id=id)
    input_rgbs = np.zeros_like(input_coors) + [255, 255, 255]

    output_coors = fetch_instance(output_coors, output_num_list, id=id)
    output_rgbs = np.zeros_like(output_coors) + [255, 0, 0]

    plot_coors = np.concatenate([input_coors, output_coors], axis=0)
    plot_rgbs = np.concatenate([input_rgbs, output_rgbs], axis=0)


    output_attrs = fetch_instance(target_attrs.numpy(), output_num_list, id=id)

    plot_points(coors=plot_coors,
                rgb=plot_rgbs,
                name='target_attrs',
                bboxes=output_attrs,
                prob=None)


