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
dimension = [100., 140., 4.]
offset = [10., 70., 3.]
anchor_size = [1.6, 3.9, 1.5]
resolution = [0.6, 0.6, 0.8]

anchor_param_list = tf.constant([[1.6, 3.9, 1.5, -1.0, 0.],
                                 [1.6, 3.9, 1.5, -1.0, np.pi/2]])

if __name__ == '__main__':
    Dataset = Dataset(task='training',
                      batch_size=batch_size,
                      config=config.aug_config,
                      num_worker=6,
                      hvd_size=1,
                      hvd_id=0)
    coors, features, num_list, input_labels = next(Dataset.train_generator())
    Dataset.stop()
    coors, num_list, idx = grid_sampling(coors, num_list, 0.6, offset=offset, dimension=dimension)
    coors = coors.numpy()
    num_list = num_list.numpy()

    voxel_idx = np.array((coors + offset) / resolution, dtype=np.int32)

    voxels = np.zeros([batch_size, int(dimension[0]/resolution[0]),  int(dimension[1]/resolution[1]),  int(dimension[2]/resolution[2]), 1])
    for b in range(batch_size):
        v = fetch_instance(voxel_idx, num_list, b)
        for i in tqdm(range(len(v))):
            voxels[b, v[i, 0], v[i, 1], v[i, 2], 0] = 1.


    id = 0
    voxel_coors, voxel_features = get_points_from_dense_voxels(voxels[id], resolution=resolution, offset=offset)
    voxel_rgbs = np.zeros_like(voxel_coors) + [255, 0, 0]

    coors = fetch_instance(coors, num_list, id)
    input_rgbs = np.zeros_like(coors) + 255

    plot_coors = np.concatenate([coors, voxel_coors], axis=0)
    plot_rgbs = np.concatenate([input_rgbs, voxel_rgbs], axis=0)

    plot_points(coors=plot_coors,
                intensity=None,
                rgb=plot_rgbs,
                name='dense_voxelization_numpy')
