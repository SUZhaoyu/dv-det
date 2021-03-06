import os

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from tqdm import tqdm
from models.tf_ops.test.test_layers import point_conv, conv_3d, fully_connected, roi_pooling, \
    get_roi_attrs_from_logits, get_bbox_attrs_from_logits

# tf.enable_eager_execution()
from models.tf_ops.custom_ops import grid_sampling, voxel_sampling, roi_logits_to_attrs, get_roi_bbox, roi_filter, voxel2col, voxel2col_grad
from models.tf_ops.test.test_utils import get_rgbs_from_coors, plot_points_from_roi_voxels, fetch_instance, plot_points
from models.utils.ops_wrapper import kernel_conv_wrapper
from data.kitti_generator import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



anchor_size = [1.6, 3.9, 1.5]
batch_size = 4
epoch = 1
if __name__ == '__main__':
    Dataset = Dataset(task='training',
                      batch_size=batch_size,
                      num_worker=6,
                      hvd_size=1,
                      hvd_id=0)
    input_coors, input_features, input_num_list, input_bbox = [], [], [], []
    for i in tqdm(range(epoch)):
        coors_d, features_d, num_list_d, bbox_d = next(Dataset.train_generator())
        input_coors.append(coors_d)
        input_features.append(features_d)
        input_num_list.append(num_list_d)
        input_bbox.append(bbox_d)
    Dataset.stop()

    # np.save("input_coors.npy", input_coors)
    # np.save("input_features.npy", input_features)
    # np.save("input_num_list.npy", input_num_list)
    # np.save("input_bbox.npy", input_bbox)

    # input_coors = np.load("input_coors.npy", allow_pickle=True)
    # input_features = get_rgbs_from_coors(input_coors)
    # input_num_list = np.load("input_num_list.npy", allow_pickle=True)
    # input_bbox = np.load("input_bbox.npy", allow_pickle=True)

    coors_p = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    features_p = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    num_list_p = tf.placeholder(dtype=tf.int32, shape=[None])
    bbox_p = tf.placeholder(dtype=tf.float32, shape=[None, 64, 9])
    coors, features, num_list, bbox = coors_p, features_p, num_list_p, bbox_p

    center_coors, center_num_list = grid_sampling(input_coors=coors,
                                    input_num_list=num_list,
                                    resolution=0.2)

    roi_attrs, roi_conf, _ = get_roi_bbox(center_coors, bbox, center_num_list, anchor_size)
    roi_conf = tf.cast(roi_conf, dtype=tf.float32)
    roi_attrs, roi_num_list = roi_filter(roi_attrs, roi_conf, center_num_list, 0.9)

    rois_attrs_noise = tf.random.uniform(shape=[tf.shape(roi_attrs)[0], 7],
                                         minval=-0.2,
                                         maxval=0.2,
                                         dtype=tf.dtypes.float32,
                                         seed=None,
                                         name=None)
    roi_attrs += rois_attrs_noise
    roi_voxels = roi_pooling(coors, features, roi_attrs, num_list, roi_num_list, voxel_size=3, padding_value=-1)
    voxels, idx = voxel2col(roi_voxels, 3)
    voxels_grad = voxel2col_grad(input_voxels=roi_voxels, output_idx=idx, output_voxels_grad=voxels)


    init_op = tf.initialize_all_variables()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.allow_soft_placement = False
    config.log_device_placement = False
    config.gpu_options.visible_device_list = '0'
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        for n in tqdm(range(epoch)):
            output_roi_voxels, output_roi_attrs, output_num_list = sess.run([voxels_grad, roi_attrs, roi_num_list],
            # output_attrs = sess.run([roi_attrs],
                                feed_dict={coors_p: input_coors[n],
                                           features_p: get_rgbs_from_coors(input_coors[n]),
                                           num_list_p: input_num_list[n],
                                           bbox_p: input_bbox[n]})

            # output_voxels = np.reshape(output_voxels, newshape=[output_voxels.shape[0], output_voxels.shape[1], 27, -1])
            # output = np.zeros([output_voxels.shape[0], output_voxels.shape[1], output_voxels.shape[3]])
            # for i in tqdm(range(output_voxels.shape[0]*output_voxels.shape[1])):
            #     input_num = i//output_voxels.shape[1]
            #     output_num = i%output_voxels.shape[1]
            #     non_void_idx = np.sum(output_voxels[input_num, output_num, :, :], axis=-1) > 0
            #     for j in range(3):
            #         output[input_num, output_num, j] = np.mean(output_voxels[input_num, output_num, non_void_idx, j])



        id = 2
        # output_voxels = fetch_instance(output, output_num_list, id=id)
        # output_roi_attrs = fetch_instance(output_roi_attrs, output_num_list, id=id)
        # plot_points_from_roi_voxels(voxels=output_voxels,
        #                             roi_attrs=output_roi_attrs,
        #                             kernel_size=3,
        #                             name='voxel2col')

        output_roi_voxels = fetch_instance(output_roi_voxels, output_num_list, id=id)
        output_roi_attrs = fetch_instance(output_roi_attrs, output_num_list, id=id)
        plot_points_from_roi_voxels(voxels=output_roi_voxels,
                                    roi_attrs=output_roi_attrs,
                                    kernel_size=3,
                                    name='voxel2col_grad')

        output_coors = fetch_instance(input_coors[n], input_num_list[n], id=id)
        output_features = fetch_instance(get_rgbs_from_coors(input_coors[n]), input_num_list[n], id=id)
        plot_points(output_coors, rgb=output_features, name='voxel2col_input')

    #
    # output_coors = fetch_instance(input_coors[i], input_num_list[i], id=id)
    # output_features = fetch_instance(get_rgbs_from_coors(input_coors[i]), input_num_list[i], id=id)
    # plot_points(output_coors, rgb=output_features, name='roi_pooling_input')

