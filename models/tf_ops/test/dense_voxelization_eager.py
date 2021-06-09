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
from models.tf_ops.loader.sampling import grid_sampling
from models.tf_ops.loader.pooling import dense_voxelization
from models.tf_ops.loader.others import anchor_iou3d, anchor_iou_filter
from models.tf_ops.loader.bbox_utils import get_roi_bbox, get_anchor_attrs, get_bev_gt_bbox
from models.tf_ops.test.test_utils import get_points_from_dense_voxels, fetch_instance, plot_points

from models.utils.iou_utils import cal_bev_iou


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 6
epoch = 2
dimension = [100., 140., 5.]
offset = [10., 70., 3.]
anchor_size = [1.6, 3.9, 1.5]
resolution = [0.6, 0.6, 0.6]

anchor_param_list = tf.constant([[1.6, 3.9, 1.5, -1.0, 0.],
                                 [1.6, 3.9, 1.5, -1.0, np.pi/2]])

if __name__ == '__main__':

    Dataset = Dataset(task='training',
                      batch_size=batch_size,
                      config=config.aug_config,
                      num_worker=6,
                      hvd_size=1,
                      hvd_id=0)
    input_coors, input_features, input_num_list, input_labels = next(Dataset.train_generator())
    Dataset.stop()
    input_coors = tf.cast(input_coors, dtype=tf.float32)

    coors, num_list, idx = grid_sampling(input_coors, input_num_list, 0.6, offset=offset, dimension=dimension)
    features = tf.gather(input_features, idx)

    voxels = dense_voxelization(input_coors=coors,
                                input_features=features,
                                input_num_list=num_list,
                                resolution=resolution,
                                dimension=dimension,
                                offset=offset)

    id = 0
    voxels = voxels.numpy()
    voxel_coors, voxel_features = get_points_from_dense_voxels(voxels[id], resolution=resolution, offset=offset)
    voxel_rgbs = np.zeros_like(voxel_coors) + [255, 0, 0]

    input_coors = fetch_instance(input_coors, input_num_list, id)
    input_rgbs = np.zeros_like(input_coors) + 255

    plot_coors = np.concatenate([input_coors, voxel_coors], axis=0)
    plot_rgbs = np.concatenate([input_rgbs, voxel_rgbs], axis=0)

    plot_points(coors=plot_coors,
                intensity=None,
                rgb=plot_rgbs,
                name='dense_voxelization')
