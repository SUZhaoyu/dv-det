import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from os.path import join
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import matplotlib.pyplot as plt
from tensorflow.python.client import timeline
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

from data.kitti_generator import Dataset
import train.kitti.kitti_config as config
# tf.enable_eager_execution()
from models.tf_ops.loader.sampling import grid_sampling, get_bev_features, bev_occupy, get_bev_anchor_point
from models.tf_ops.loader.pooling import bev_projection
from models.tf_ops.loader.others import anchor_iou3d, anchor_iou_filter
from models.tf_ops.loader.bbox_utils import get_roi_bbox, get_anchor_attrs, get_bev_gt_bbox
from models.tf_ops.test.test_utils import fetch_instance, plot_points

from models.utils.iou_utils import cal_bev_iou


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 6
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
    input_coors, input_features, input_num_list, input_labels = next(Dataset.train_generator())
    Dataset.stop()

    coors = tf.cast(input_coors, dtype=tf.float32)
    coors, num_list, idx = grid_sampling(coors, input_num_list, 0.8, offset=offset, dimension=dimension)

    bev_occupy_maps = bev_occupy(input_coors=coors,
                                 input_num_list=num_list,
                                 resolution=0.6,
                                 dimension=dimension,
                                 offset=offset)

    anchor_coors, anchor_num_list = get_bev_anchor_point(bev_occupy=bev_occupy_maps,
                                                         resolution=0.6,
                                                         kernel_resolution=1.6,
                                                         offset=offset,
                                                         height=[-0.5, -1.])



    coors = coors.numpy()
    anchor_coors = anchor_coors.numpy()
    num_list = num_list.numpy()
    anchor_num_list = anchor_num_list.numpy()
    bev_occupy_maps = bev_occupy_maps.numpy()

    print(anchor_coors.shape)

    id = 5
    input_coors = fetch_instance(input_coors, input_num_list, id=id)
    input_rgbs = np.zeros_like(input_coors) + [255, 255, 255]

    coors = fetch_instance(coors, num_list, id=id)
    rgbs = np.zeros_like(coors) + [0, 0, 255]

    anchor_coors = fetch_instance(anchor_coors, anchor_num_list, id=id)
    anchor_rgbs = np.zeros_like(anchor_coors) + [255, 0, 0]

    plot_coors = np.concatenate([input_coors, coors, anchor_coors], axis=0)
    plot_rgbs = np.concatenate([input_rgbs, rgbs, anchor_rgbs], axis=0)

    plt.imsave(join('/home/tan/tony/threejs/html', 'bev_anchor_coors.png'), bev_occupy_maps[id, :, :, 0])

    plot_points(coors=plot_coors,
                rgb=plot_rgbs,
                name='bev_anchor_coors')


