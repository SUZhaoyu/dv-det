import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import horovod.tensorflow as hvd
import os
from tqdm import tqdm
from os.path import join, dirname
import sys
import argparse
from shutil import rmtree, copyfile

HOME = join(dirname(os.getcwd()))
sys.path.append(HOME)

from models import kitti_model as model
from train.kitti import kitti_config as config
from data.kitti_generator import Dataset
from train.train_utils import get_train_op, get_config, save_best_sess, set_training_controls
from models.tf_ops.loader.others import rotated_nms3d_idx

hvd.init()
is_hvd_root = hvd.rank() == 0

bn = hvd.SyncBatchNormalization()

if config.local:
    log_dir = '/home/tan/tony/dv-det/checkpoints/local-debug'
    try:
        rmtree(log_dir)
    except:
        pass
    os.mkdir(log_dir)

else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', dest='log_dir', default='test')
    args = parser.parse_args()
    log_dir = args.log_dir

if is_hvd_root:
    copyfile(config.config_dir, join(log_dir, config.config_dir.split('/')[-1]))

DatasetTrain = Dataset(task="training",
                       batch_size=config.batch_size,
                       config=config.aug_config,
                       num_worker=config.num_worker,
                       hvd_size=hvd.size(),
                       hvd_id=hvd.rank())
# DatasetTrain.stop()

DatasetValid = Dataset(task="validation",
                       validation=True,
                       batch_size=config.batch_size,
                       hvd_size=hvd.size(),
                       hvd_id=hvd.rank())
# DatasetValid.stop()

training_batch = DatasetTrain.batch_sum
validation_batch = DatasetValid.batch_sum
decay_batch = training_batch * config.decay_epochs

input_coors_p, input_features_p, input_num_list_p, input_bbox_p = \
    model.stage1_inputs_placeholder(input_channels=1,
                                    bbox_padding=config.bbox_padding)
is_training_p = tf.placeholder(dtype=tf.bool, shape=[], name="stage1_training")

step = tf.Variable(0, name='stage1_step')


lr, bn, wd = set_training_controls(config=config,
                                   lr=config.init_lr_stage1,
                                   scale=config.lr_scale_stage1,
                                   decay_batch=decay_batch,
                                   step=step,
                                   hvd_size=hvd.size(),
                                   prefix='end2end')

coors, features, num_list, roi_coors, roi_attrs, roi_conf_logits, roi_num_list = \
    model.stage1_model(input_coors=input_coors_p,
                       input_features=input_features_p,
                       input_num_list=input_num_list_p,
                       is_training=is_training_p,
                       trainable=True,
                       is_eval=False,
                       mem_saving=True,
                       bn=bn)

stage1_loss, averaged_roi_iou = model.stage1_loss(roi_coors=roi_coors,
                                                  pred_roi_attrs=roi_attrs,
                                                  roi_conf_logits=roi_conf_logits,
                                                  roi_num_list=roi_num_list,
                                                  bbox_labels=input_bbox_p,
                                                  wd=wd)

roi_conf = tf.nn.sigmoid(roi_conf_logits)
nms_idx = rotated_nms3d_idx(roi_attrs, roi_conf, nms_overlap_thresh=0.8, nms_conf_thres=0.1)
roi_coors = tf.gather(roi_coors, nms_idx, axis=0)
roi_attrs = tf.gather(roi_attrs, nms_idx, axis=0)
roi_conf_logits = tf.gather(roi_conf_logits, nms_idx, axis=0)
roi_num_list = tf.expand_dims(tf.shape(nms_idx)[0], axis=0)

roi_ious = model.get_roi_iou(roi_coors=roi_coors,
                             pred_roi_attrs=roi_attrs,
                             roi_num_list=roi_num_list,
                             bbox_labels=input_bbox_p)

bbox_attrs, bbox_conf_logits, bbox_dir_logits, bbox_num_list, bbox_idx = \
    model.stage2_model(coors=coors,
                       features=features,
                       num_list=num_list,
                       roi_attrs=roi_attrs,
                       roi_conf_logits=roi_conf_logits,
                       roi_ious=roi_ious,
                       roi_num_list=roi_num_list,
                       is_training=is_training_p,
                       trainable=True,
                       is_eval=False,
                       mem_saving=True,
                       bn=bn)

bbox_conf = tf.nn.sigmoid(bbox_conf_logits)


stage2_loss, averaged_bbox_iou = model.stage2_loss(roi_attrs=roi_attrs,
                                                   pred_bbox_attrs=bbox_attrs,
                                                   bbox_conf_logits=bbox_conf_logits,
                                                   bbox_dir_logits=bbox_dir_logits,
                                                   bbox_num_list=bbox_num_list,
                                                   bbox_labels=input_bbox_p,
                                                   bbox_idx=bbox_idx,
                                                   roi_ious=roi_ious,
                                                   wd=wd)

total_loss = stage1_loss + stage2_loss

train_op = get_train_op(total_loss, lr, opt='adam', global_step=step, use_hvd=True)
stage1_train_op = get_train_op(stage1_loss, stage1_lr, var_keywords=['stage1'], opt='adam', global_step=stage1_step, use_hvd=True)
stage2_train_op = get_train_op(stage2_loss, stage2_lr, var_keywords=['stage2'], opt='adam', global_step=stage2_step, use_hvd=True)

tf_summary = tf.summary.merge_all()
hooks = [hvd.BroadcastGlobalVariablesHook(0)]
session_config = get_config(gpu=config.gpu_list[hvd.rank()])

training_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'))
validation_writer = tf.summary.FileWriter(os.path.join(log_dir, 'valid'))
saver = tf.train.Saver(max_to_keep=None)

def train_one_epoch(sess, step, dataset_generator, writer):
    iou_sum = 0
    iter = tqdm(range(training_batch)) if is_hvd_root else range(training_batch)
    for _ in iter:
        coors, features, num_list, bboxes = next(dataset_generator)
        iou, _, summary = sess.run([averaged_bbox_iou, train_op, tf_summary],
                                    feed_dict={input_coors_p: coors,
                                               input_features_p: features,
                                               input_num_list_p: num_list,
                                               input_bbox_p: bboxes,
                                               is_training_p: True})

        iou_sum += iou
        step += 1
        if is_hvd_root:
            writer.add_summary(summary, step)

    iou = iou_sum / training_batch

    if is_hvd_root:
        summary = tf.Summary()
        summary.value.add(tag='Overall IoU', simple_value=iou)
        writer.add_summary(summary, step)
        print("Training: Total IoU={:0.2f}".format(iou))
    return step


def valid_one_epoch(sess, step, dataset_generator, writer):
    iou_sum = 0
    instance_count = 0
    iter = tqdm(range(validation_batch)) if is_hvd_root else range(validation_batch)
    for _, batch_id in enumerate(iter):
        coors, features, num_list, bboxes = next(dataset_generator)
        iou, summary = sess.run([averaged_bbox_iou, tf_summary],
                                feed_dict={input_coors_p: coors,
                                           input_features_p: features,
                                           input_num_list_p: num_list,
                                           input_bbox_p: bboxes,
                                           is_training_p: False})

        iou_sum += (iou * len(features))
        instance_count += len(features)
        if is_hvd_root:
            writer.add_summary(summary, step + batch_id)

    iou = iou_sum / instance_count

    if is_hvd_root:
        summary = tf.Summary()
        summary.value.add(tag='Overall IoU', simple_value=iou)
        writer.add_summary(summary, step)
        print("Validation: Total IoU={:0.2f}".format(iou))
    return iou



def main():
    with tf.train.MonitoredTrainingSession(hooks=hooks, config=session_config) as mon_sess:
        train_generator = DatasetTrain.train_generator()
        valid_generator = DatasetValid.valid_generator()
        best_result = 0.
        step = 0
        # valid_one_epoch(mon_sess, step, valid_generator, validation_writer)

        for epoch in range(config.total_epoch):
            if is_hvd_root:
                print("Epoch: {}".format(epoch))
            step = train_one_epoch(mon_sess, step, train_generator, training_writer)
            if epoch % config.valid_interval == 0:  # and EPOCH != 0:
                result = valid_one_epoch(mon_sess, step, valid_generator, validation_writer)
                if is_hvd_root:
                    best_result = save_best_sess(mon_sess, best_result, result,
                                                 log_dir, saver, replace=True, log=is_hvd_root, inverse=False,
                                                 save_anyway=False)


if __name__ == '__main__':
    main()
