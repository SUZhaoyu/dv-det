import os

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from tqdm import tqdm

from data.kitti_generator import Dataset
# tf.enable_eager_execution()
from models.tf_ops.custom_ops import grid_sampling
from models.tf_ops.test.test_utils import fetch_instance, plot_points

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 16
epoch = 50
if __name__ == '__main__':
    Dataset = Dataset(task='training',
                      batch_size=batch_size,
                      num_worker=6,
                      hvd_size=1,
                      hvd_id=0)
    input_coors, input_features, input_num_list = [], [], []
    for i in tqdm(range(epoch)):
        coors_d, features_d, num_list_d, _ = next(Dataset.valid_generator())
        features_d = np.zeros_like(coors_d)
        accu = 0
        for j in range(len(num_list_d)):
            features_d[int(accu):int(accu + num_list_d[j]), :] += np.random.randint(low=0, high=255, size=3).astype(
                np.float32)
            accu += num_list_d[j]
        input_coors.append(coors_d)
        input_num_list.append(num_list_d)
        input_features.append(features_d)
    Dataset.stop()

    coors_p = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    features_p = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    num_list_p = tf.placeholder(dtype=tf.int32, shape=[None])
    coors, features, num_list = coors_p, features_p, num_list_p
    # coors, features, num_list = point_conv(coors, features, num_list,  16,0.1, 'layer_0')
    # coors, features, num_list = point_conv(coors, features, num_list,  32,0.2, 'layer_2')
    # coors, features, num_list = point_conv(coors, features, num_list,  64,0.3, 'layer_4')
    # coors, features, num_list = point_conv(coors, features, num_list, 128,0.4, 'layer_6')
    coors, features, num_list, idx = grid_sampling(coors, features, num_list, 0.1)
    coors, features, num_list, idx = grid_sampling(coors, features, num_list, 0.2)
    coors, features, num_list, idx = grid_sampling(coors, features, num_list, 0.4)
    coors, features, num_list, idx = grid_sampling(coors, features, num_list, 0.8)
    # coors, features, num_list = grid_sampling(coors, features, num_list, 0.1)
    # coors, features, num_list = grid_sampling(coors, features, num_list, 0.1)
    # coors, num_list, idx = grid_sampling(coors, num_list, 0.1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.allow_soft_placement = False
    config.log_device_placement = False
    config.gpu_options.visible_device_list = '0'
    init_op = tf.initialize_all_variables()
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        for i in tqdm(range(epoch)):
            output_coors, output_features, output_num_list, output_idx = sess.run([coors, features, num_list, idx],
                                                                                  feed_dict={coors_p: input_coors[i],
                                                                                             features_p: input_features[
                                                                                                 i],
                                                                                             num_list_p: input_num_list[
                                                                                                 i]},
                                                                                  options=run_options,
                                                                                  run_metadata=run_metadata)
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline_grid_sampling_{}.json'.format(i), 'w') as f:
                f.write(ctf)

            print("finished.")

            # print(i, num_list_d, output_centers.shape, output_num_list, np.sum(output_num_list))

    id = 0
    # output_voxels = fetch_instance(output_voxels, output_num_list, id=id)
    output_coors = fetch_instance(output_coors, output_num_list, id=id)
    output_features = fetch_instance(output_features, output_num_list, id=id)
    output_idx = fetch_instance(output_idx, output_num_list, id=id)
    output_idx = np.expand_dims(np.sort(output_idx), axis=-1)
    plot_points(coors=output_coors,
                rgb=output_features,
                name='grid_sampling_2')
    # plot_points_from_voxels(voxels=output_voxels,
    #                         center_coors=output_centers,
    #                         resolution=0.1,
    #                         name='voxel_sample')
    # plot_points_from_voxels_with_color(voxels=output_voxels,
    #                         center_coors=output_centers,
    #                         resolution=0.1,
    #                         name='voxel_sample_rgb')
