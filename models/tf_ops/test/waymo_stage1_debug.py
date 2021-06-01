import tensorflow as tf
import sys
sys.path.append("/home/tan/tony/dv-det")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import horovod.tensorflow as hvd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tqdm import tqdm
from os.path import join, dirname
import sys
import argparse
from shutil import rmtree, copyfile
from models.tf_ops.loader.sampling import grid_sampling, grid_sampling_thrust, voxel_sampling_idx_binary

HOME = join(dirname(os.getcwd()))
sys.path.append(HOME)

from models import waymo_model_cls_reg as model
from train.waymo import waymo_config as config
from data.waymo_generator import Dataset
from train.train_utils import get_train_op, get_config, save_best_sess, set_training_controls

DatasetTrain = Dataset(task="train",
                       batch_size=config.batch_size,
                       config=config.aug_config,
                       num_worker=config.num_worker,
                       hvd_size=8,
                       hvd_id=0)

input_coors_p, input_features_p, input_num_list_p, input_bbox_p = \
    model.stage1_inputs_placeholder(input_channels=2,
                                    bbox_padding=config.bbox_padding)
is_stage1_training_p = tf.placeholder(dtype=tf.bool, shape=[], name="stage1_training")


# coors, num_list, _ = grid_sampling(input_coors_p, input_num_list_p, 0.1, offset=config.offset_training, dimension=config.dimension_training)
#
# voxel_idx, _, features = voxel_sampling_idx_binary(input_coors=input_coors_p,
#                                                    input_features=input_features_p,
#                                                    input_num_list=num_list,
#                                                    center_coors=coors,
#                                                    center_num_list=num_list,
#                                                    resolution=0.1,
#                                                    dimension=config.dimension_training,
#                                                    offset=config.offset_training,
#                                                    grid_buffer_size=3,
#                                                    output_pooling_size=5)

# coors, features, num_list, roi_coors, roi_logits, roi_conf_logits, roi_attrs, roi_num_list = \
coors = \
    model.stage1_model(input_coors=input_coors_p,
                       input_features=input_features_p,
                       input_num_list=input_num_list_p,
                       is_training=is_stage1_training_p,
                       is_eval=False,
                       trainable=True,
                       mem_saving=True,
                       bn=1.)

# stage1_loss, averaged_roi_iou = model.stage1_loss(roi_coors=roi_coors,
#                                                   pred_roi_logits=roi_logits,
#                                                   roi_conf_logits=roi_conf_logits,
#                                                   roi_num_list=roi_num_list,
#                                                   bbox_labels=input_bbox_p,
#                                                   wd=stage1_wd)

init_op = tf.initialize_all_variables()

def main():
    with tf.Session() as sess:
        sess.run(init_op)
        input_coors, input_features, input_num_list, input_bboxes = next(DatasetTrain.train_generator())
        DatasetTrain.stop()
        output = sess.run(coors, feed_dict={input_coors_p: input_coors,
                                            input_features_p: input_features,
                                            input_num_list_p: input_num_list,
                                            input_bbox_p: input_bboxes,
                                            is_stage1_training_p: True})


if __name__ == '__main__':
    main()
