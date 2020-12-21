import numpy as np
import tensorflow as tf
from point_viz.converter import PointvizConverter

from data_utils.pc_generator_back_up import KITTI_PC as GENERATOR
from tf_ops.custom_ops_loader import la_roi_pooling, farthest_point_sample, \
    gather_foreground_points, get_roi_bbox, la_roi_pooling_grad_test
from data_utils.normalization import convert_threejs_coors, convert_threejs_bbox

from tf_ops.ops_test.test_utils import get_rgbs_from_coors, voxel2points, get_rgbs_concatenation
from models.iou_utils import get_roi_attrs_from_logits

NPOINT = 20000
BATCH_SIZE = 12
BBOX_PADDING = 64
IDX = 6
ANCHOR_SIZE = [1.6, 3.9, 1.5]
ANCHOR_SIZE_TF = tf.constant(ANCHOR_SIZE)
Dataset = GENERATOR(phase='training',
                    npoint=NPOINT,
                    batch_size=BATCH_SIZE,
                    validation=True)
Generator = Dataset.valid_generator(start_idx=1888-IDX)

input_coors_p = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
input_features_p = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
input_attentions_p = tf.placeholder(dtype=tf.int32, shape=[None, None, 1])
input_bbox_p = tf.placeholder(dtype=tf.float32, shape=[None, BBOX_PADDING, 9])

coors, features, num_list = gather_foreground_points(input_coors=input_coors_p,
                                                     input_features=input_features_p,
                                                     input_foreground_points=input_attentions_p)

center_coors, center_num_list = farthest_point_sample(input_coors=coors,
                                                      input_num_list=num_list,
                                                      reduce_factor=0.05)

roi_attrs, roi_conf = get_roi_bbox(input_coors=center_coors,
                                     bboxes=input_bbox_p,
                                     input_num_list=center_num_list,
                                     anchor_size=ANCHOR_SIZE_TF,
                                     expand_ratio=0.1,
                                     diff_thres=3)

rois_attrs_noise = tf.random.uniform(shape=[tf.shape(roi_attrs)[0], 7],
                                minval=-0.5,
                                maxval=0.5,
                                dtype=tf.dtypes.float32,
                                seed=None,
                                name=None)
roi_attrs += rois_attrs_noise

voxels, idxs, weights = la_roi_pooling(input_coors=coors,
                                        input_features=features,
                                        voxel_size=5,
                                        roi_attrs=roi_attrs,
                                        input_num_list=num_list,
                                        rois_num_list=center_num_list)

grad = la_roi_pooling_grad_test(input_features=features,
                                output_idx=idxs,
                                output_weights=weights,
                                grad=voxels)


if __name__ == '__main__':
    with tf.Session() as sess:
        batch_coors, _, batch_attentions, batch_bbox = next(Generator)
        foreground_coors, foreground_num_list, output_coors, output_voxels, output_num_list, output_rois, output_conf, input_grad = sess.run(
            [coors, num_list, center_coors, voxels, center_num_list, roi_attrs, roi_conf, grad],
            feed_dict={input_coors_p: batch_coors,
                       input_features_p: get_rgbs_from_coors(batch_coors),
                       input_attentions_p: batch_attentions,
                       input_bbox_p: batch_bbox})

        total_coors = batch_coors[IDX]
        total_rgbs = get_rgbs_from_coors(batch_coors)[IDX]
        accu_num_list = np.cumsum(output_num_list)
        foreground_accu_num_list = np.cumsum(foreground_num_list)
        output_voxels = output_voxels[accu_num_list[IDX-1]:accu_num_list[IDX], ...]
        output_coors = output_coors[accu_num_list[IDX-1]:accu_num_list[IDX], ...]
        output_rois = output_rois[accu_num_list[IDX-1]:accu_num_list[IDX], ...]
        output_conf = output_conf[accu_num_list[IDX-1]:accu_num_list[IDX]]
        foreground_coors = foreground_coors[foreground_accu_num_list[IDX-1]:foreground_accu_num_list[IDX], ...]
        input_grad = input_grad[foreground_accu_num_list[IDX-1]:foreground_accu_num_list[IDX], ...]
        # output_rois[:, :3] *= ANCHOR_SIZE

        output_points = voxel2points(voxels=output_voxels,
                                     center_coors=output_coors,
                                     roi_dims=output_rois,
                                     roi_conf=output_conf,
                                     kernel_size=5,
                                     use_self_RGB=True)

        output_coors, output_rgbs = get_rgbs_concatenation(coors_list=[total_coors, output_coors],
                                                           color_list=[[255., 255., 255.], [255., 0., 0.]])

        Converter = PointvizConverter(home='/media/data1/threejs')
        Converter.compile(task_name="roi_pooling_output",
                          coors=convert_threejs_coors(output_points[..., :3]),
                          default_rgb=output_points[..., 3:6])
        Converter.compile(task_name="roi_pooling_input",
                          coors=convert_threejs_coors(total_coors),
                          default_rgb=total_rgbs)
        Converter.compile(task_name="roi_pooling_input_grad",
                          coors=convert_threejs_coors(foreground_coors),
                          default_rgb=input_grad)
        Converter.compile(task_name="roi_pooling_foreground",
                          coors=convert_threejs_coors(output_coors),
                          default_rgb=output_rgbs)
