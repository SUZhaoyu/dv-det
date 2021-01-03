import os

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from tqdm import tqdm
from data.kitti_generator import Dataset

# tf.enable_eager_execution()
from models.tf_ops.custom_ops import grid_sampling, voxel_sampling, voxel_sampling_binary, grid_sampling_thrust
from models.utils.ops_wrapper import kernel_conv_wrapper
from models.tf_ops.test.test_utils import fetch_instance, get_rgbs_from_coors, plot_points_from_voxels_with_color

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


batch_size = 16
epoch = 10
if __name__ == '__main__':
    Dataset = Dataset(task='training',
                      batch_size=batch_size,
                      num_worker=6,
                      hvd_size=1,
                      hvd_id=0)
    input_coors, input_features, input_num_list = [], [], []
    for i in tqdm(range(epoch)):
        coors_d, _, num_list_d, _ = next(Dataset.train_generator())
        input_coors.append(coors_d)
        input_num_list.append(num_list_d)
    Dataset.stop()
    # np.save("input_coors.npy", input_coors)
    # np.save("input_features.npy", input_features)
    # np.save("input_num_list.npy", input_num_list)

    # input_coors = np.load("input_coors.npy", allow_pickle=True)
    # input_features = np.load("input_features.npy", allow_pickle=True)
    # input_num_list = np.load("input_num_list.npy", allow_pickle=True)

    coors_p = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    features_p = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    num_list_p = tf.placeholder(dtype=tf.int32, shape=[None])
    # coors, num_list = coors_p, num_list_p

    # coors, features, num_list, voxels = point_sampling(coors, features, num_list, 16,0.8, 'layer_0')
    coors_0, num_list_0 = grid_sampling_thrust(coors_p, num_list_p, 0.1)
    coors_1, num_list_1 = grid_sampling_thrust(coors_0, num_list_0, 0.2)
    coors_2, num_list_2 = grid_sampling_thrust(coors_1, num_list_1, 0.4)
    coors_3, num_list_3 = grid_sampling_thrust(coors_2, num_list_2, 0.6)
    coors_4, num_list_4 = grid_sampling_thrust(coors_3, num_list_3, 0.8)
    voxels, idx = voxel_sampling_binary(input_coors=coors_3,
                                        input_features=tf.ones_like(coors_1) * 255,
                                        input_num_list=num_list_3,
                                        center_coors=coors_4,
                                        center_num_list=num_list_4,
                                        resolution=0.2,
                                        padding=-1,
                                        dimension=[70.4, 80.0, 4.0],
                                        offset=[0., 40.0, 3.0])

    init_op = tf.initialize_all_variables()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.allow_soft_placement = False
    config.log_device_placement = False
    config.gpu_options.visible_device_list = '0'
    with tf.Session(config=config) as sess:
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        for i in tqdm(range(epoch)):
            # coors_d, features_d, num_list_d, _ = next(Dataset.train_generator())
            # output_features, output_centers, output_num_list, output_voxels = sess.run([features, coors, num_list, voxels],
            output_centers, output_num_list, output_features, output_idx = sess.run([coors_2, num_list_2, voxels, idx],
                                                                        # output_voxels = sess.run(voxels,
                                                                        feed_dict={coors_p: input_coors[i],
                                                                                   features_p: get_rgbs_from_coors(input_coors[i]),
                                                                                   num_list_p: input_num_list[i]})
                                                                        # options=run_options,
                                                                        # run_metadata=run_metadata)

            # print(output_centers.shape)
            ## time.sleep(0.1)
            #
            # tl = timeline.Timeline(run_metadata.step_stats)
            # ctf = tl.generate_chrome_trace_format(show_memory=True)
            # with open('timeline_{}.json'.format(i), 'w') as f:
            #     f.write(ctf)

            # print(i, num_list_d, output_centers.shape, output_num_list, np.sum(output_num_list))
    for i in tqdm(range(output_idx.shape[0])):
        for j in range(27):
            if output_idx[i, j, 0] > 1e6:
                print(i, j, output_idx[i, j, 0])

    id = 6
    output_voxels = fetch_instance(output_features, output_num_list, id=id)
    output_centers = fetch_instance(output_centers, output_num_list, id=id)
    # plot_points_from_voxels(voxels=output_voxels,
    #                         center_coors=output_centers,
    #                         resolution=0.1,
    #                         name='voxel_sample')
    plot_points_from_voxels_with_color(voxels=output_voxels,
                            center_coors=output_centers,
                            resolution=0.2,
                                       self_rgbs=True,
                            name='voxel_sampling_binary')
    #