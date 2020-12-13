import os

import tensorflow as tf
from tensorflow.python.client import timeline

from data.kitti_generator import Dataset
# tf.enable_eager_execution()
from models.tf_ops.custom_ops import grid_down_sample
from models.tf_ops.test.test_utils import fetch_instance, plot_points_with_intensity

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    Dataset = Dataset(num_worker=1,
                      hvd_size=5,
                      hvd_id=2)

    input_coors_p = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    input_features_p = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    input_num_list_p = tf.placeholder(dtype=tf.int32, shape=[None])

    coors, num_list = grid_down_sample(input_coors=input_coors_p,
                                       input_num_list=input_num_list_p,
                                       resolution=0.1)
    coors, num_list = grid_down_sample(input_coors=coors,
                                       input_num_list=num_list,
                                       resolution=0.2)
    coors, num_list = grid_down_sample(input_coors=coors,
                                       input_num_list=num_list,
                                       resolution=0.3)
    # coors, num_list = grid_down_sample(input_coors=coors,
    #                                    input_num_list=num_list,
    #                                    resolution=0.4)
    # coors, num_list = grid_down_sample(input_coors=coors,
    #                                    input_num_list=num_list,
    #                                    resolution=0.5)

    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        input_coors, input_features, input_num_list, input_bboxes = next(Dataset.train_generator())
        Dataset.stop()
        output_coors, output_num_list = sess.run([coors, num_list],
                                                 feed_dict={input_coors_p: input_coors,
                                                            input_features_p: input_features[..., :1],
                                                            input_num_list_p: input_num_list},
                                                 options=run_options,
                                                 run_metadata=run_metadata)
        print(input_num_list)
        print(output_num_list)
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_thrust.json', 'w') as f:
            f.write(ctf)

        print("finished.")

    id = 9
    output_coors = fetch_instance(output_coors, output_num_list, id=id)
    plot_points_with_intensity(coors=output_coors,
                               intensity=None,
                               name='grid_down_sample')
