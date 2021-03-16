import os

import tensorflow as tf
from tqdm import tqdm

from data.waymo_generator import Dataset
# tf.enable_eager_execution()
import train.waymo.waymo_config as config
from models.tf_ops.loader.sampling import grid_sampling
from models.tf_ops.loader.bbox_utils import get_roi_bbox
from models.tf_ops.custom_ops import la_roi_pooling_fast, la_roi_pooling_fast_grad
from models.tf_ops.loader.others import roi_filter
from models.tf_ops.test.test_utils import get_rgbs_from_coors, plot_points_from_roi_voxels, fetch_instance, plot_points

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



anchor_size = [1.6, 3.9, 1.5]
batch_size = 4
epoch = 2


if __name__ == '__main__':
    WaymoDataset = Dataset(task='train',
                           batch_size=batch_size,
                           num_worker=6,
                           hvd_size=1,
                           hvd_id=0)
    input_coors, input_features, input_num_list, input_bbox = [], [], [], []
    for i in tqdm(range(epoch)):
        coors_d, features_d, num_list_d, bbox_d = next(WaymoDataset.train_generator())
        input_coors.append(coors_d)
        input_features.append(features_d)
        input_num_list.append(num_list_d)
        input_bbox.append(bbox_d)
    WaymoDataset.stop()

    coors_p = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    features_p = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    num_list_p = tf.placeholder(dtype=tf.int32, shape=[None])
    bbox_p = tf.placeholder(dtype=tf.float32, shape=[None, 256, 9])
    coors, features, num_list, bbox = coors_p, features_p, num_list_p, bbox_p

    coors, num_list, _ = grid_sampling(input_coors=coors,
                                       input_num_list=num_list,
                                       resolution=0.6,
                                       dimension=config.dimension_training,
                                       offset=config.offset_training)

    center_coors, center_num_list, _ = grid_sampling(input_coors=coors,
                                                     input_num_list=num_list,
                                                     resolution=5.,
                                                     dimension=config.dimension_training,
                                                     offset=config.offset_training)

    roi_attrs, roi_conf, _ = get_roi_bbox(center_coors, bbox, center_num_list, anchor_size)
    roi_conf = tf.cast(roi_conf, dtype=tf.float32)
    roi_attrs, roi_num_list, _ = roi_filter(input_roi_attrs=roi_attrs,
                                            input_roi_conf=roi_conf,
                                            input_roi_ious=roi_conf,
                                            input_num_list=center_num_list,
                                            conf_thres=0.9,
                                            iou_thres=0,
                                            max_length=512,
                                            with_negative=False)

    rois_attrs_noise = tf.random.uniform(shape=[tf.shape(roi_attrs)[0], 7],
                                         minval=-0.1,
                                         maxval=0.1,
                                         dtype=tf.dtypes.float32,
                                         seed=None,
                                         name=None)
    roi_attrs += rois_attrs_noise


    voxels, idx, weight = la_roi_pooling_fast(input_coors=coors,
                                             input_features=features,
                                             roi_attrs=roi_attrs,
                                             input_num_list=num_list,
                                             roi_num_list=roi_num_list,
                                             dimension=config.dimension_training,
                                             offset=config.offset_training,
                                             grid_buffer_resolution=2.0,
                                             grid_buffer_size=16,
                                             voxel_size=5,
                                             padding_value=0.,
                                             pooling_size=8)

    grad = la_roi_pooling_fast_grad(features, idx, weight, voxels)


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
                                                                  feed_dict={coors_p: input_coors[i],
                                                                             features_p: get_rgbs_from_coors(
                                                                                 input_coors[i]),
                                                                             num_list_p: input_num_list[i],
                                                                             bbox_p: input_bbox[i]})

    id = 2
    output_coors = fetch_instance(output_coors, output_num_list, id=id)
    output_grad = fetch_instance(output_grad, output_num_list, id=id)
    plot_points(output_coors, rgb=output_grad, name='roi_pooling_grad')

    output_coors = fetch_instance(input_coors[i], input_num_list[i], id=id)
    output_features = fetch_instance(get_rgbs_from_coors(input_coors[i], repeat=20), input_num_list[i], id=id)
    plot_points(output_coors, rgb=output_features, name='roi_pooling_input')
