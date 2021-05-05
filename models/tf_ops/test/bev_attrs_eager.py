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
    coors, num_list, idx = grid_sampling(coors, num_list, 0.1, offset=offset, dimension=dimension)
    features = tf.gather(features, idx)

    bev_img = bev_projection(input_coors=coors,
                             input_features=features,
                             input_num_list=num_list,
                             resolution=0.4,
                             dimension=dimension,
                             offset=offset,
                             buffer_size=10)

    bev_coors, bev_features, bev_num_list = get_bev_features(bev_img=bev_img,
                                                             resolution=0.4,
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
                                                cls_thres=0)

    anchor_attrs = get_anchor_attrs(anchor_coors=bev_coors,
                                    anchor_param_list=anchor_param_list)

    gt_attrs = tf.reshape(gt_attrs, [-1, tf.shape(gt_attrs)[2]])
    anchor_attrs = tf.reshape(anchor_attrs, [-1, tf.shape(anchor_attrs)[2]])
    gt_conf = tf.reshape(gt_conf, [-1])
    gt_idx = tf.reshape(gt_idx, [-1])

    idx = tf.squeeze(tf.where(tf.greater_equal(gt_conf, 1)))
    gt_attrs = tf.gather(gt_attrs, idx)
    gt_idx = tf.gather(gt_idx, idx)
    anchor_attrs = tf.gather(anchor_attrs, idx)



    bev_iou = cal_bev_iou(gt_attrs, anchor_attrs)
    gt_conf = anchor_iou_filter(bev_iou, gt_idx, labels, gt_conf, idx)
    gt_conf = tf.gather(gt_conf, idx)
    bev_iou = tf.cast(gt_conf, dtype=tf.float32) * bev_iou




    input_coors = coors
    output_coors = np.concatenate([bev_coors.numpy(), np.zeros([bev_coors.numpy().shape[0], 1]) + -1.0], axis=-1)
    bev_num_list = bev_num_list.numpy()

    id = 0
    input_coors = fetch_instance(coors, num_list, id=id)
    input_rgbs = np.zeros_like(input_coors) + [255, 255, 255]


    output_coors = fetch_instance(output_coors, num_list, id=id)
    output_rgbs = np.zeros_like(output_coors) + [255, 0, 0]

    plot_coors = np.concatenate([input_coors, output_coors], axis=0)
    plot_rgbs = np.concatenate([input_rgbs, output_rgbs], axis=0)

    # output_attrs = fetch_instance(gt_attrs.numpy(), bev_num_list*2, id=id)
    # bev_iou = fetch_instance(bev_iou.numpy(), bev_num_list*2, id=id)
    gt_attrs = gt_attrs.numpy()
    anchor_attrs = anchor_attrs.numpy()
    bev_iou = bev_iou.numpy()
    thres = 0.3
    print(np.sum(bev_iou > thres))
    gt_attrs = gt_attrs[bev_iou > thres, :]
    anchor_attrs = anchor_attrs[bev_iou > thres, :]
    bev_iou = bev_iou[bev_iou > thres]

    start = 20
    span = 1

    # gt_attrs = gt_attrs[start:start+span, :]
    # anchor_attrs = anchor_attrs[start:start+span, :]
    output_attrs = np.concatenate([gt_attrs, anchor_attrs], axis=0)

    plot_points(coors=input_coors,
                rgb=input_rgbs,
                name='bev_coors_iou_filter',
                bboxes=anchor_attrs,
                prob=bev_iou)


