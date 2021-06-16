import os
from os.path import join
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

CWD = os.path.dirname(os.path.abspath(__file__))

def get_bev_anchor_coors(bev_img, resolution, offset):
    offset = np.array(offset, dtype=np.float32)
    resolution = np.array(resolution, dtype=np.float32)
    bev_img_shape = bev_img.get_shape()
    img_w = bev_img_shape[1]
    img_l = bev_img_shape[2]

    bev_idx = tf.range(img_w * img_l)  # [w*l]
    anchor_coors = tf.cast(tf.stack([bev_idx // img_l, bev_idx % img_l], axis=-1), dtype=tf.float32)  # [w*l, 2] -> [x, y]
    anchor_coors = anchor_coors * resolution[0:2] + resolution[0:2] / 2. - tf.expand_dims(offset[0:2], axis=0)
    # bev_z_coors = tf.zeros(shape=[img_w * img_l, 1]) + height  # [w * l, 1]
    # bev_3d_coors = tf.expand_dims(tf.concat([bev_2d_coors, bev_z_coors], axis=-1), axis=0)  # [1, w*l, 3]
    # bev_3d_coors = tf.expand_dims(bev_2d_coors, axis=0)  # [1, w*l, 2]
    # anchor_coors = tf.tile(bev_3d_coors, [batch_size, 1, 1])

    return anchor_coors


def get_anchors(anchor_coors, anchor_params, batch_size):
    # length = 17360
    length = anchor_coors.get_shape()[0]#.value()
    output_anchor = []
    num_anchor = len(anchor_params)
    for anchor_param in anchor_params: # [w, l, h, z, r]
        w = tf.ones([length, 1]) * anchor_param[0]
        l = tf.ones([length, 1]) * anchor_param[1]
        h = tf.ones([length, 1]) * anchor_param[2]

        x = tf.ones([length, 1]) * anchor_coors[:, 0:1]
        y = tf.ones([length, 1]) * anchor_coors[:, 1:2]
        z = tf.ones([length, 1]) * anchor_param[3]

        r = tf.ones([length, 1]) * anchor_param[4]

        anchor = tf.concat([w, l, h, x, y, z, r], axis=-1) #[w*l, 7]
        output_anchor.append(anchor)

    output_anchor = tf.stack(output_anchor, axis=1, name='anchor_stack') #[w*l, 2, 7]
    output_anchor = tf.reshape(output_anchor, shape=[length * num_anchor, 7]) #[w*l*2, 7]
    output_anchor = tf.expand_dims(output_anchor, axis=0) #[1, w*l*2, 7]
    output_anchor = tf.tile(output_anchor, [batch_size, 1, 1]) # [n, w*l*2, 7]
    # anchor_num_list = tf.ones([batch_size], dtype=tf.int32) * num_anchor * length

    return output_anchor


iou_utils_gpu_exe = tf.load_op_library(join(CWD, '../build', 'nms.so'))
def get_anchor_iou(anchor_bbox, label_bbox, ignore_height=True):
    ious = iou_utils_gpu_exe.boxes_iou_op(anchor_bbox, label_bbox, ignore_height)
    return ious


