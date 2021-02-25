import os
from os.path import join
import numpy as np
import tensorflow as tf
from point_viz.converter import PointvizConverter
from tensorflow.python.client import timeline
from tqdm import tqdm
# from models.tf_ops.loader.others import rotated_nms3d_idx
from data.utils.normalization import convert_threejs_coors, convert_threejs_bbox_with_colors
import horovod.tensorflow as hvd
from models.tf_ops.loader.bbox_utils import get_roi_bbox
from models.utils.iou_utils import cal_3d_iou
from models.utils.loss_utils import get_masked_average
hvd.init()

os.system("rm -r {}".format('/home/tan/tony/threejs/waymo-stage1'))
Converter = PointvizConverter(home='/home/tan/tony/threejs/waymo-stage1')

from models import waymo_model as model
from train.configs import waymo_config as config
from data.waymo_generator import Dataset

model_path = '/home/tan/tony/dv-det/checkpoints/waymo-stage1-avg_pool/test/best_model_0.6606513755109455' # 68.8%@non-mem-saving
data_home = '/home/tan/tony/dv-det/eval/waymo/data'
visualization = True

DatasetTrain = Dataset(task="train",
                       batch_size=4,
                       config=config.aug_config,
                       num_worker=config.num_worker,
                       hvd_size=hvd.size(),
                       hvd_id=hvd.rank())
dataset_generator = DatasetTrain.train_generator()


input_coors_p, input_features_p, input_num_list_p, input_bbox_p = model.stage1_inputs_placeholder(input_channels=2)
is_training_p = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')


coors, features, num_list, roi_coors, roi_attrs, roi_conf_logits, roi_num_list = \
    model.stage1_model(input_coors=input_coors_p,
                       input_features=input_features_p,
                       input_num_list=input_num_list_p,
                       is_training=is_training_p,
                       is_eval=True,
                       trainable=False,
                       mem_saving=True,
                       bn=1.)

roi_conf = tf.nn.sigmoid(roi_conf_logits)


stage1_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stage1')
saver = tf.train.Saver(stage1_variables)
config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0"
config.gpu_options.allow_growth = True
config.allow_soft_placement = False
config.log_device_placement = False


if __name__ == '__main__':
    with tf.Session(config=config) as sess:
        saver.restore(sess, model_path)
        prediction_output = []
        overall_iou = []
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        for _ in tqdm(range(2)):
            batch_input_coors, batch_input_features, batch_input_num_list, batch_input_bboxes = \
                next(dataset_generator)
            output_bboxes, output_coors, output_conf, output_num_list = \
                sess.run([roi_attrs, roi_coors, roi_conf, roi_num_list],
                         feed_dict={input_coors_p: batch_input_coors,
                                    input_features_p: batch_input_features,
                                    input_num_list_p: batch_input_num_list,
                                    input_bbox_p: batch_input_bboxes,
                                    is_training_p: False})

            print(output_coors.shape)
            print(output_num_list)
