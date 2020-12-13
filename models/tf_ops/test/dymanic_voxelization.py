import os

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from tqdm import tqdm

# tf.enable_eager_execution()
from models.tf_ops.custom_ops import grid_sampling, voxel_sampling
from models.utils.ops_wrapper import kernel_conv_wrapper

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# def point_conv(input_coors, input_features, input_num_list, c_out, resolution, scope):
#     coors, num_list = grid_sampling(input_coors=input_coors,
#                                       input_num_list=input_num_list,
#                                       resolution=resolution)
#
#     voxels = dynamic_voxelization(input_coors=input_coors,
#                                   input_features=input_features,
#                                   input_num_list=input_num_list,
#                                   center_coors=coors,
#                                   center_num_list=num_list,
#                                   resolution=resolution,
#                                   padding=0.)
#
#     features = kernel_conv_wrapper(inputs=voxels,
#                                    num_output_channels=c_out,
#                                    scope=scope)
#     return coors, features, num_list

def point_sampling(input_coors, input_features, input_num_list, c_out, resolution, scope):
    coors, num_list = grid_sampling(input_coors=input_coors,
                                    input_num_list=input_num_list,
                                    resolution=resolution * 2)

    voxels = voxel_sampling(input_coors=input_coors,
                            input_features=input_features,
                            input_num_list=input_num_list,
                            center_coors=coors,
                            center_num_list=num_list,
                            resolution=resolution,
                            padding=-1)

    features = kernel_conv_wrapper(inputs=voxels,
                                   num_output_channels=c_out,
                                   scope=scope,
                                   bn_decay=1.)

    # features = tf.reduce_mean(voxels, axis=1)

    return coors, num_list, features, voxels


batch_size = 1
epoch = 1000
if __name__ == '__main__':
    # Dataset = Dataset(task='training',
    #                   batch_size=batch_size,
    #                   num_worker=6,
    #                   hvd_size=1,
    #                   hvd_id=0)
    # input_coors, input_features, input_num_list = [], [], []
    # for i in tqdm(range(epoch)):
    #     coors_d, features_d, num_list_d, _ = next(Dataset.train_generator())
    #     input_coors.append(coors_d)
    #     input_features.append(features_d)
    #     input_num_list.append(num_list_d)
    # Dataset.stop()
    # np.save("input_coors.npy", input_coors)
    # np.save("input_features.npy", input_features)
    # np.save("input_num_list.npy", input_num_list)

    input_coors = np.load("input_coors.npy", allow_pickle=True)
    input_features = np.load("input_features.npy", allow_pickle=True)
    input_num_list = np.load("input_num_list.npy", allow_pickle=True)

    coors_p = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    features_p = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    num_list_p = tf.placeholder(dtype=tf.int32, shape=[None])
    coors, features, num_list = coors_p, features_p, num_list_p

    # coors, features, num_list, voxels = point_sampling(coors, features, num_list, 16,0.8, 'layer_0')
    coors, num_list, features, _ = point_sampling(coors, features, num_list, 16, 0.1, 'layer_0')
    coors, num_list, features, _ = point_sampling(coors, features, num_list, 32, 0.2, 'layer_2')
    coors, num_list, features, _ = point_sampling(coors, features, num_list, 64, 0.4, 'layer_4')
    coors, num_list, features, _ = point_sampling(coors, features, num_list, 128, 0.8, 'layer_6')
    coors, num_list, features, voxels = point_sampling(coors, features, num_list, 128, 1.0, 'layer_8')
    # coors, features, num_list, _ = point_sampling(coors, features, num_list, 64,0.4, 'layer_4')
    # coors, features, num_list, voxels = point_sampling(coors, features, num_list,128,0.8, 'layer_6')
    # center_coors, center_num_list = grid_sampling(input_coors=coors,
    #                                 input_num_list=num_list,
    #                                 resolution=0.1)
    # voxels = voxel_sampling(input_coors=coors,
    #                         input_features=features,
    #                         input_num_list=num_list,
    #                         center_coors=center_coors,
    #                         center_num_list=center_num_list,
    #                         resolution=0.1,
    #                         padding=-1)

    # coors, num_list = grid_sampling(input_coors=coors_p,
    #                                 input_num_list=num_list_p,
    #                                 resolution=0.2)
    #
    # voxels = voxel_sampling(input_coors=coors_p,
    #                         input_features=features_p,
    #                         input_num_list=num_list_p,
    #                         center_coors=coors,
    #                         center_num_list=num_list,
    #                         resolution=0.1,
    #                         padding=-1)

    init_op = tf.initialize_all_variables()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.allow_soft_placement = False
    config.log_device_placement = False
    config.gpu_options.visible_device_list = '0'
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        for i in tqdm(range(epoch)):
            # coors_d, features_d, num_list_d, _ = next(Dataset.train_generator())
            # output_features, output_centers, output_num_list, output_voxels = sess.run([features, coors, num_list, voxels],
            output_centers, output_num_list, output_features = sess.run([coors, num_list, voxels],
                                                                        # output_voxels = sess.run(voxels,
                                                                        feed_dict={coors_p: input_coors[i],
                                                                                   features_p: input_features[i],
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

    # id = 6
    # output_voxels = fetch_instance(output_features, output_num_list, id=id)
    # output_centers = fetch_instance(output_centers, output_num_list, id=id)
    # # plot_points_from_voxels(voxels=output_voxels,
    # #                         center_coors=output_centers,
    # #                         resolution=0.1,
    # #                         name='voxel_sample')
    # plot_points_from_voxels_with_color(voxels=output_voxels,
    #                         center_coors=output_centers,
    #                         resolution=0.1,
    #                         name='voxel_sample_rgb_new')
    #
