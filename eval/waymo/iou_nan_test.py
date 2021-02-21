import os
from os.path import join
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
from models.utils.iou_utils import cal_3d_iou


data_home = '/home/tan/tony/dv-det/eval/waymo/data'

if __name__ == '__main__':

    gt_roi_attrs = np.load(join(data_home, 'gt_roi_attrs_nan.npy'), allow_pickle=True)
    pred_roi_attrs = np.load(join(data_home, 'pred_roi_attrs_nan.npy'), allow_pickle=True)
    ious = np.load(join(data_home, 'roi_ious_nan.npy'), allow_pickle=True)

    gt_roi_attrs_p = tf.placeholder(dtype=tf.float32, shape=[None, 7])
    pred_roi_attrs_p = tf.placeholder(dtype=tf.float32, shape=[None, 7])

    # for _ in range(10000):
    roi_ious = cal_3d_iou(gt_attrs=gt_roi_attrs_p, pred_attrs=pred_roi_attrs_p, clip=False)
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "0"
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = False
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        output_ious = sess.run(roi_ious, feed_dict={gt_roi_attrs_p: gt_roi_attrs,
                                                    pred_roi_attrs_p: pred_roi_attrs})
    # print(roi_ious[74978])

    print(" ")
