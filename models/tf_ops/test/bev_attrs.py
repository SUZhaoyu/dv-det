import os
from os.path import join
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.kitti_generator import Dataset
import train.kitti.kitti_config as config
# tf.enable_eager_execution()
from models.tf_ops.loader.sampling import grid_sampling, get_bev_features
from models.tf_ops.loader.pooling import bev_projection
from models.tf_ops.loader.bbox_utils import get_roi_bbox, get_anchor_attrs
from models.tf_ops.test.test_utils import fetch_instance, plot_points


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 8
epoch = 2
dimension = [100., 140., 9.]
offset = [10., 70., 5.]
anchor_size = [1.6, 3.9, 1.5]

anchor_param_list = tf.constant([[1.6, 3.9, 1.5, -1.0, 0],
                                 [1.6, 3.9, 1.5, -1.0, np.pi / 2]])

if __name__ == '__main__':
    Dataset = Dataset(task='training',
                      batch_size=batch_size,
                      config=config.aug_config,
                      num_worker=6,
                      hvd_size=1,
                      hvd_id=0)
    input_coors, input_features, input_num_list, input_labels = [], [], [], []
    for i in tqdm(range(epoch)):
        coors_d, features_d, num_list_d, labels_d = next(Dataset.train_generator())
        accu = 0
        features_d = np.random.randn(len(coors_d), 128) + 1.
        for j in range(len(num_list_d)):
            # features_d[int(accu):int(accu + num_list_d[j]), :] += np.random.randint(low=0, high=255, size=3).astype(
            #     np.float32)
            accu += num_list_d[j]
        input_coors.append(coors_d)
        input_num_list.append(num_list_d)
        input_features.append(features_d)
        input_labels.append(labels_d)

    Dataset.stop()
    coors_p = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    features_p = tf.placeholder(dtype=tf.float32, shape=[None, 128])
    num_list_p = tf.placeholder(dtype=tf.int32, shape=[batch_size])
    labels_p = tf.placeholder(dtype=tf.float32, shape=[batch_size, 128, 9])
    coors, features, num_list = coors_p, features_p, num_list_p
    coors, num_list, idx = grid_sampling(coors, num_list, 0.4, offset=offset, dimension=dimension)
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
                                                             offset=offset,
                                                             z_base_coor=-1.0)

    gt_roi_attrs, gt_roi_conf, gt_roi_diff = get_roi_bbox(input_coors=bev_coors,
                                                          bboxes=labels_p,
                                                          input_num_list=bev_num_list,
                                                          anchor_size=anchor_size,
                                                          expand_ratio=0.2,
                                                          diff_thres=config.diff_thres,
                                                          cls_thres=config.cls_thres)

    anchor_attrs = get_anchor_attrs(anchor_coors=bev_coors,
                                    anchor_param_list=anchor_param_list)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.allow_soft_placement = False
    config.log_device_placement = False
    config.gpu_options.visible_device_list = '0'
    init_op = tf.initialize_all_variables()
    with tf.Session(config=config) as sess:
        # sess.run(init_op)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        for i in tqdm(range(epoch)):
            output_coors, output_num_list, output_img, output_attrs = sess.run([bev_coors, bev_num_list, bev_img, anchor_attrs],
                                                      feed_dict={coors_p: input_coors[i],
                                                                 features_p: input_features[i],
                                                                 num_list_p: input_num_list[i],
                                                                 labels_p: input_labels[i]},
                                                      options=run_options,
                                                      run_metadata=run_metadata)
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('bev_proj_{}.json'.format(i), 'w') as f:
                f.write(ctf)

            print("finished.")

    id = 4
    input_coors = fetch_instance(input_coors[i], input_num_list[i], id=id)
    input_rgbs = np.zeros_like(input_coors) + [255, 255, 255]

    output_coors = fetch_instance(output_coors, output_num_list, id=id)
    output_rgbs = np.zeros_like(output_coors) + [255, 0, 0]

    plot_coors = np.concatenate([input_coors, output_coors], axis=0)
    plot_rgbs = np.concatenate([input_rgbs, output_rgbs], axis=0)

    output_attrs = fetch_instance(output_attrs, output_num_list, id=id)
    output_attrs = output_attrs[output_attrs[:, 0] > 0.2, :]


    plot_points(coors=plot_coors,
                rgb=plot_rgbs,
                name='bev_coors',
                bboxes=output_attrs)

    # output_img = np.sum(output_idx >= 0, axis=-1)
    plt.imsave(join('/home/tan/tony/threejs/html', "bev_img.png"), output_img[id, :, :, 0])
