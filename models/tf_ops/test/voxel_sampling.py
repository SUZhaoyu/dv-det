import os

import tensorflow as tf
from tqdm import tqdm
from tensorflow.python.client import timeline
from data.kitti_generator import Dataset
# tf.enable_eager_execution()
from models.tf_ops.loader.sampling import grid_sampling_thrust, voxel_sampling_feature, voxel_sampling_idx, voxel_sampling_idx_binary
# from models.utils.ops_wrapper import kernel_conv_wrapper
from models.tf_ops.test.test_utils import fetch_instance, get_rgbs_from_coors, plot_points_from_voxels_with_color, \
    get_rgbs_from_coors_tf
from data.utils.normalization import convert_threejs_coors
from point_viz.converter import PointvizConverter
Converter = PointvizConverter("/home/tan/tony/threejs")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


batch_size = 1
epoch = 10
if __name__ == '__main__':
    KittiDataset = Dataset(task='validation',
                           batch_size=batch_size,
                           num_worker=6,
                           hvd_size=1,
                           hvd_id=0,
                           validation=True)
    input_coors, input_features, input_num_list = [], [], []
    for i in tqdm(range(epoch)):
        coors_d, _, num_list_d, _ = next(KittiDataset.valid_generator())
        input_coors.append(coors_d)
        input_num_list.append(num_list_d)
    KittiDataset.stop()
    # np.save("input_coors.npy", input_coors)
    # np.save("input_features.npy", input_features)
    # np.save("input_num_list.npy", input_num_list)

    # input_coors = np.load("input_coors.npy", allow_pickle=True)
    # input_features = np.load("input_features.npy", allow_pickle=True)
    # input_num_list = np.load("input_num_list.npy", allow_pickle=True)

    coors_p = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    features_p = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    num_list_p = tf.placeholder(dtype=tf.int32, shape=[None])
    coors, num_list = coors_p, num_list_p

    # coors, features, num_list, voxels = point_sampling(coors, features, num_list, 16,0.8, 'layer_0')
    coors_0, num_list_0, _ = grid_sampling_thrust(coors_p, num_list_p, 0.2, dimension=[100, 160.0, 8.0], offset=[10., 80.0, 4.0])
    # coors_1, num_list_1, _ = grid_sampling_thrust(coors_0, num_list_0, 0.2, dimension=[100, 160.0, 8.0], offset=[10., 80.0, 4.0])
    # coors_2, num_list_2, _ = grid_sampling_thrust(coors_1, num_list_1, 0.4, dimension=[100, 160.0, 8.0], offset=[10., 80.0, 4.0])


    # voxel_idx, _, features = voxel_sampling_idx(input_coors=coors,
    #                                          input_features=get_rgbs_from_coors_tf(coors),
    #                                          input_num_list=num_list,
    #                                          center_coors=coors_0,
    #                                          center_num_list=num_list_0,
    #                                          resolution=0.2,
    #                                          dimension=[100, 160.0, 8.0],
    #                                          offset=[10., 80.0, 4.0],
    #                                          grid_buffer_size=3,
    #                                          output_pooling_size=5)

    voxel_idx, output_weight, features = voxel_sampling_idx_binary(input_coors=coors,
                                                       input_features=get_rgbs_from_coors_tf(coors),
                                                       input_num_list=num_list,
                                                       center_coors=coors_0,
                                                       center_num_list=num_list_0,
                                                       resolution=0.1,
                                                       dimension=[100, 160.0, 8.0],
                                                       offset=[10., 80.0, 4.0],
                                                       grid_buffer_size=3,
                                                       output_pooling_size=5)

    voxels = voxel_sampling_feature(input_features=features,
                                    output_idx=voxel_idx,
                                    output_weight=output_weight,
                                    padding=0)

    # voxels = voxel_sampling_binary(input_coors=coors_1,
    #                                     input_features=get_rgbs_from_coors_tf(coors_1),
    #                                     input_num_list=num_list_1,
    #                                     center_coors=coors_2,
    #                                     center_num_list=num_list_2,
    #                                     resolution=0.2,
    #                                     padding=-1,
    #                                     dimension=[100, 160.0, 8.0],
    #                                     offset=[10., 80.0, 4.0])

    init_op = tf.initialize_all_variables()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.allow_soft_placement = False
    config.log_device_placement = False
    config.gpu_options.visible_device_list = '0'
    with tf.Session(config=config) as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        for i in tqdm(range(epoch)):
            # coors_d, features_d, num_list_d, _ = next(Dataset.train_generator())
            # output_features, output_centers, output_num_list, output_voxels = sess.run([features, coors, num_list, voxels],
            output_centers, output_num_list, output_features, output_idx = sess.run([coors_0, num_list_0, voxels, voxel_idx],
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
            # with open('voxel_sampling_idx.json'.format(i), 'w') as f:
            #     f.write(ctf)

            # print(i, num_list_d, output_centers.shape, output_num_list, np.sum(output_num_list))
    # for i in tqdm(range(output_idx.shape[0])):
    #     for j in range(27):
    #         if output_idx[i, j, 0] > 1e6:
    #             print(i, j, output_idx[i, j, 0])

    id = 0

    raw_coors = fetch_instance(input_coors[i], input_num_list[i], id=id)
    features = fetch_instance(get_rgbs_from_coors(input_coors[i]), input_num_list[i], id=id)

    Converter.compile(coors=convert_threejs_coors(raw_coors),
                      default_rgb=features,
                      task_name='voxel_sampling_input')


    output_voxels = fetch_instance(output_features, output_num_list, id=id)
    output_centers = fetch_instance(output_centers, output_num_list, id=id)

    plot_points_from_voxels_with_color(voxels=output_voxels,
                                       center_coors=output_centers,
                                       resolution=0.1,
                                       mask=0,
                                       self_rgbs=True,
                                       name='voxel_sampling_testing')
    #
