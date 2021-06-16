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
from models.tf_ops.loader.others import anchor_iou_filter
from models.tf_ops.loader.anchor_utils import get_bev_anchor_coors, get_anchors, get_anchor_iou
from models.tf_ops.loader.bbox_utils import get_roi_bbox, get_anchor_attrs, get_bev_gt_bbox
from models.tf_ops.test.test_utils import fetch_instance, plot_points
from models.utils.model_blocks import bev_compression


from models.utils.iou_utils import cal_bev_iou


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 6
epoch = 2
resolution = [2.0, 2.0, 0.8]
id = 3

dimension_params = {'dimension': config.dimension_training,
                    'offset': config.offset_training}


if __name__ == '__main__':
    Dataset = Dataset(task='training',
                      batch_size=batch_size,
                      config=config.aug_config,
                      num_worker=6,
                      hvd_size=1,
                      hvd_id=0)
    input_coors, input_features, input_num_list, labels = next(Dataset.train_generator())
    Dataset.stop()

    coors = tf.cast(input_coors, dtype=tf.float32)
    coors, num_list, idx = grid_sampling(coors, input_num_list, 0.6, offset=dimension_params['offset'], dimension=dimension_params['dimension'])
    features = tf.gather(input_features, idx)

    bev_img = bev_compression(input_coors=coors,
                              input_features=features,
                              input_num_list=num_list,
                              resolution=resolution,
                              dimension_params=dimension_params)

    anchor_coors = get_bev_anchor_coors(bev_img, resolution, dimension_params['offset'])
    anchor = get_anchors(anchor_coors, config.anchor_params, batch_size)
    anchor_ious = get_anchor_iou(anchor[id, :, :7], labels[id, :, :7], True)

    input_coors = coors.numpy()
    input_points = fetch_instance(input_coors, input_num_list, id)
    anchor_attrs = anchor.numpy()[id]
    label_attrs = labels[id]

    plot_points(coors=input_points,
                rgb=None,
                name='bev_anchors',
                bboxes=label_attrs,
                prob=None)
