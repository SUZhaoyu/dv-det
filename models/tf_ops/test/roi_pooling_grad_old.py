import os

import tensorflow as tf
from tqdm import tqdm

from data.kitti_generator import Dataset
# tf.enable_eager_execution()
from models.tf_ops.custom_ops import grid_sampling, get_roi_bbox, roi_filter, la_roi_pooling_fast, la_roi_pooling_fast_grad
from models.tf_ops.test.test_utils import get_rgbs_from_coors, fetch_instance, plot_points

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



anchor_size = [1.6, 3.9, 1.5]
batch_size = 4
epoch = 2
if __name__ == '__main__':
    Dataset = Dataset(task='validation',
                      batch_size=batch_size,
                      num_worker=6,
                      hvd_size=1,
                      hvd_id=0)
    input_coors, input_features, input_num_list, input_bbox = [], [], [], []
    for i in tqdm(range(epoch)):
        coors_d, features_d, num_list_d, bbox_d = next(Dataset.valid_generator())
        input_coors.append(coors_d)
        input_features.append(features_d)
        input_num_list.append(num_list_d)
        input_bbox.append(bbox_d)
    Dataset.stop()

    coors_p = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    features_p = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    num_list_p = tf.placeholder(dtype=tf.int32, shape=[None])
    bbox_p = tf.placeholder(dtype=tf.float32, shape=[None, 64, 9])
    coors, features, num_list, bbox = coors_p, features_p, num_list_p, bbox_p

    # coors, num_list = grid_sampling(input_coors=coors,
    #                                 input_num_list=num_list,
    #                                 resolution=0.2)

    center_coors, center_num_list = grid_sampling(input_coors=coors,
                                                  input_num_list=num_list,
                                                  resolution=0.8)

    roi_attrs, roi_conf, _ = get_roi_bbox(center_coors, bbox, center_num_list, anchor_size)
    roi_conf = tf.cast(roi_conf, dtype=tf.float32)
    roi_attrs, roi_num_list, _ = roi_filter(roi_attrs, roi_conf, center_num_list, 0.9, 0, False)

    rois_attrs_noise = tf.random.uniform(shape=[tf.shape(roi_attrs)[0], 7],
                                         minval=-0.2,
                                         maxval=0.2,
                                         dtype=tf.dtypes.float32,
                                         seed=None,
                                         name=None)
    roi_attrs += rois_attrs_noise
    voxels, idx, weight = la_roi_pooling_fast(coors, features, roi_attrs, num_list, roi_num_list, voxel_size=5, padding_value=-1, pooling_size=8)
    grad = la_roi_pooling_grad_test(voxels, features, idx, weight)

    init_op = tf.initialize_all_variables()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.allow_soft_placement = False
    config.log_device_placement = False
    config.gpu_options.visible_device_list = '0'
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        for i in tqdm(range(epoch)):
            output_coors, output_grad, output_num_list = sess.run([coors, grad, num_list],
                                                                  # output_attrs = sess.run([roi_attrs],
                                                                  feed_dict={coors_p: input_coors[i],
                                                                             features_p: get_rgbs_from_coors(
                                                                                 input_coors[i]),
                                                                             num_list_p: input_num_list[i],
                                                                             bbox_p: input_bbox[i]})

    id = 1
    output_coors = fetch_instance(output_coors, output_num_list, id=id)
    output_grad = fetch_instance(output_grad, output_num_list, id=id)
    plot_points(output_coors, rgb=output_grad, name='roi_pooling_grad')

