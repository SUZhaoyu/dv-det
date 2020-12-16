from models.tf_ops.custom_ops import roi_filter
import numpy as np
import tensorflow as tf

# tf.enable_eager_execution()

conf_p = tf.placeholder(shape=[None], dtype=tf.float32)
num_list_p = tf.placeholder(shape=[None], dtype=tf.int32)

if __name__ == '__main__':
    conf = np.random.rand(100).astype(np.float32)
    num_list = np.ones(10).astype(np.int32) * 10
    list, idx = roi_filter(input_roi_conf=conf_p,
                                  input_num_list=num_list_p,
                                  conf_thres=0.5)
    with tf.Session() as sess:
        output_list, output_idx = sess.run([list, idx], feed_dict={conf_p: conf, num_list_p: num_list})
    print(output_list, output_idx)
