import os

import tensorflow as tf
from tqdm import tqdm

import train.configs.rcnn_config as training_config
from data.kitti_generator import Dataset
# tf.enable_eager_execution()
from models.utils.model_layers import point_conv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    Dataset = Dataset(task='training',
                      batch_size=16,
                      num_worker=3,
                      hvd_size=5,
                      hvd_id=2)

    input_coors_p = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    input_features_p = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    input_num_list_p = tf.placeholder(dtype=tf.int32, shape=[None])
    is_training_p = tf.placeholder(dtype=tf.bool, shape=[])
    model_params = {'xavier': training_config.xavier,
                    'stddev': training_config.stddev,
                    'activation': training_config.activation,
                    'dimension': training_config.dimension,
                    'offset': training_config.offset}
    base_params = training_config.base_params

    coors, features, num_list = input_coors_p, input_features_p, input_num_list_p
    for layer_name in sorted(base_params.keys()):
        coors, features, num_list = point_conv(input_coors=coors,
                                               input_features=features,
                                               input_num_list=num_list,
                                               layer_params=base_params[layer_name],
                                               scope=layer_name,
                                               is_training=is_training_p,
                                               model_params=model_params,
                                               bn_decay=1.)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    config.gpu_options.visible_device_list = '0'
    init_op = tf.initialize_all_variables()
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        for i in tqdm(range(1000)):
            input_coors, input_features, input_num_list, input_bboxes = next(Dataset.train_generator())
            output_coors, output_features, output_num_list = sess.run([coors, features, num_list],
                                                                      feed_dict={input_coors_p: input_coors,
                                                                                 input_features_p: input_features[...,
                                                                                                   :1],
                                                                                 input_num_list_p: input_num_list,
                                                                                 is_training_p: True})

            # tl = timeline.Timeline(run_metadata.step_stats)
            # ctf = tl.generate_chrome_trace_format()
            # with open('roi_inference.json', 'w') as f:
            #     f.write(ctf)

        print("finished.")
