import os
from os.path import join

import tensorflow as tf
from tqdm import tqdm

from data.kitti_generator import Dataset
# tf.enable_eager_execution()
from models.tf_ops.loader.sampling import grid_sampling_thrust, voxel_sampling_idx_binary, voxel_sampling_feature, voxel_sampling_feature_grad_test, voxel_sampling_idx
from models.tf_ops.test.test_utils import fetch_instance, get_rgbs_from_coors, plot_points, get_rgbs_from_coors_tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


batch_size = 16
epoch = 10
CWD = '/home/tan/tony/dv-det/models/tf_ops'

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

    # coors, features, num_list, voxels = point_sampling(coors, features, num_list, 16,0.8, 'layer_0')
    coors_0, num_list_0, _ = grid_sampling_thrust(coors_p, num_list_p, 0.2, dimension=[120, 160.0, 8.0], offset=[20., 80.0, 5.0])
    coors_1, num_list_1, _ = grid_sampling_thrust(coors_0, num_list_0, 0.4, dimension=[120, 160.0, 8.0], offset=[20., 80.0, 5.0])
    coors_2, num_list_2, _ = grid_sampling_thrust(coors_1, num_list_1, 0.6, dimension=[120, 160.0, 8.0], offset=[20., 80.0, 5.0])

    voxel_idx, features, sorted_coors = voxel_sampling_idx_binary(input_coors=coors_1,
                                                                    input_features=get_rgbs_from_coors_tf(coors_1),
                                                                    input_num_list=num_list_1,
                                                                    center_coors=coors_2,
                                                                    center_num_list=num_list_2,
                                                                    resolution=0.2,
                                                                    dimension=[100, 160.0, 8.0],
                                                                    offset=[10., 80.0, 4.0],
                                                                    grid_buffer_size=3,
                                                                    output_pooling_size=5)

    voxels = voxel_sampling_feature(input_features=features,
                                    output_idx=voxel_idx,
                                    padding=0)



    #
    input_grad = voxel_sampling_feature_grad_test(input_features=features,
                                                output_idx=voxel_idx,
                                                grad=voxels)

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
            output_coors, output_num_list, output_features, layer_input_coors, layer_input_features, layer_input_num_list = \
                sess.run([sorted_coors, num_list_1, input_grad, coors_1, get_rgbs_from_coors_tf(coors_1), num_list_1],
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

    id = 6
    output_features = fetch_instance(output_features, output_num_list, id=id)
    output_coors = fetch_instance(output_coors, output_num_list, id=id)
    raw_input_coors = fetch_instance(layer_input_coors, layer_input_num_list, id=id)
    raw_input_features = fetch_instance(layer_input_features, layer_input_num_list, id=id)
    plot_points(coors=output_coors,
                rgb=output_features,
                name='voxel_sampling_binary_grad')
    plot_points(coors=raw_input_coors,
                rgb=raw_input_features,
                name='voxel_sampling_binary_input')
    #
