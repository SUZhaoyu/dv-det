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
from models.tf_ops.loader.sampling import grid_sampling
from models.tf_ops.loader.pooling import bev_projection
from models.tf_ops.test.test_utils import fetch_instance, plot_points


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 8
epoch = 2
dimension = [100., 140., 9.]
offset = [10., 70., 5.]

if __name__ == '__main__':
    Dataset = Dataset(task='training',
                      batch_size=batch_size,
                      config=config.aug_config,
                      num_worker=6,
                      hvd_size=1,
                      hvd_id=0)
    input_coors, input_features, input_num_list = [], [], []
    for i in tqdm(range(epoch)):
        coors_d, features_d, num_list_d, _ = next(Dataset.train_generator())
        accu = 0
        features_d = np.random.randn(len(coors_d), 128) + 1.
        for j in range(len(num_list_d)):
            # features_d[int(accu):int(accu + num_list_d[j]), :] += np.random.randint(low=0, high=255, size=3).astype(
            #     np.float32)
            accu += num_list_d[j]
        input_coors.append(coors_d)
        input_num_list.append(num_list_d)
        input_features.append(features_d)
    Dataset.stop()

    coors_p = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    features_p = tf.placeholder(dtype=tf.float32, shape=[None, 128])
    num_list_p = tf.placeholder(dtype=tf.int32, shape=[None])
    coors, features, num_list = coors_p, features_p, num_list_p
    coors, num_list, idx = grid_sampling(coors, num_list, 0.6, offset=offset, dimension=dimension)
    features = tf.gather(features, idx)
    bev_img = bev_projection(input_coors=coors,
                             input_features=features,
                             input_num_list=num_list,
                             resolution=0.6,
                             dimension=dimension,
                             offset=offset,
                             buffer_size=8)


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
            output_coors, output_num_list, output_img = sess.run([coors, num_list, bev_img],
                                                      feed_dict={coors_p: input_coors[i],
                                                                 features_p: input_features[i],
                                                                 num_list_p: input_num_list[i]},
                                                      options=run_options,
                                                      run_metadata=run_metadata)
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('bev_proj_{}.json'.format(i), 'w') as f:
                f.write(ctf)

            print("finished.")

    id = 4
    # output_coors = fetch_instance(output_coors, output_num_list, id=id)
    # plot_points(coors=output_coors,
    #             name='bev_projection')

    # output_img = np.sum(output_idx >= 0, axis=-1)
    plt.imsave(join('/home/tan/tony/threejs/html', "bev_img.png"), output_img[id, :, :, 0])
