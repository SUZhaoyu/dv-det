import numpy as np
import tensorflow as tf

from models import kitti_model as model
from data.kitti_generator import Dataset
from train.kitti import kitti_config as config


DatasetTrain = Dataset(task="training",
                       batch_size=config.batch_size,
                       config=config.aug_config,
                       num_worker=config.num_worker,
                       hvd_size=1,
                       hvd_id=0)

input_coors_p, input_features_p, input_num_list_p, input_bbox_p = \
    model.stage1_inputs_placeholder(input_channels=1,
                                    bbox_padding=config.bbox_padding)
is_stage1_training_p = tf.placeholder(dtype=tf.bool, shape=[], name="stage1_training")

stage1_step = tf.Variable(0, name='stage1_step')
stage1_loader = tf.train.Saver()

coors, features, num_list, roi_coors, roi_attrs, roi_conf_logits, roi_num_list = \
    model.stage1_model(input_coors=input_coors_p,
                       input_features=input_features_p,
                       input_num_list=input_num_list_p,
                       is_training=is_stage1_training_p,
                       is_eval=False,
                       trainable=True,
                       mem_saving=True,
                       bn=1.)