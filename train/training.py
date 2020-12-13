from __future__ import division

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import horovod.tensorflow as hvd
import os
from tqdm import tqdm
from os.path import join, dirname
import sys
import argparse
from shutil import copyfile, rmtree

HOME = join(dirname(os.getcwd()))
sys.path.append(HOME)

from models import rcnn as model
from train.configs import roi_config as config
from data.kitti_generator import Dataset
from train.train_utils import get_bn_decay, get_learning_rate, get_train_op, get_config, get_weight_decay, \
    save_best_sess

hvd.init()
is_hvd_root = hvd.rank() == 0

if config.local:
    log_dir = '/home/tan/tony/dv_detection/checkpoints/bbox_RoI'
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
                       config=config.aug_config,
                       num_worker=config.num_worker,
                       hvd_size=hvd.size(),
                       hvd_id=hvd.rank())

DatasetValid = Dataset(task="validation",
                       hvd_size=hvd.size(),
                       hvd_id=hvd.rank())

training_batch = DatasetTrain.batch_sum
validation_batch = DatasetValid.batch_sum
decay_batch = training_batch * config.decay_epochs

input_coors_p, input_features_p, input_num_list_p, input_bbox_p = \
    model.inputs_placeholder(input_channels=1,
                             bbox_padding=config.bbox_padding)
is_training_p = tf.placeholder(dtype=tf.bool, shape=[], name="roi_training")
global_step = tf.train.get_or_create_global_step()

roi_lr = get_learning_rate(init_lr=config.init_lr,
                           current_step=global_step,
                           decay_step=decay_batch,
                           decay_rate=config.lr_decay,
                           name='roi_learning_rate',
                           hvd_size=hvd.size(),
                           lr_scale=config.lr_scale)

roi_bn = get_bn_decay(init_decay=0.5,
                      current_step=global_step,
                      decay_step=decay_batch,
                      decay_rate=0.5,
                      name="roi_bn_decay")

wd = get_weight_decay(init_decay=config.weight_decay,
                      current_step=global_step,
                      decay_step=decay_batch,
                      decay_rate=0.8)

base_coors, roi_attrs, roi_conf_logits, num_list = \
    model.model(input_coors=input_coors_p,
                input_features=input_features_p,
                input_num_list=input_num_list_p,
                is_training=is_training_p,
                config=config,
                bn=roi_bn)

roi_loss, roi_iou = \
    model.get_loss(base_coors=base_coors,
                   pred_roi_attrs=roi_attrs,
                   pred_roi_conf_logits=roi_conf_logits,
                   num_list=num_list,
                   bbox_labels=input_bbox_p,
                   wd=wd)

train_op = get_train_op(roi_loss, roi_lr, opt='adam', global_step=global_step, use_hvd=True)
summary = tf.summary.merge_all()
hooks = [hvd.BroadcastGlobalVariablesHook(0),
         tf.train.StopAtStepHook(config.training_epochs * training_batch)]
session_config = get_config(gpu=config.gpu_list[hvd.rank()])

training_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'))
validation_writer = tf.summary.FileWriter(os.path.join(log_dir, 'valid'))

saver = tf.train.Saver(max_to_keep=None)

vars = {'global_step': global_step,
        'input_coors_p': input_coors_p,
        'input_features_p': input_features_p,
        'input_num_list_p': input_num_list_p,
        'input_bbox_p': input_bbox_p,
        'is_training_p': is_training_p,
        'training_batch': training_batch,
        'validation_batch': validation_batch,
        'iou': roi_iou,
        'train_op': train_op,
        'summary': summary}


def train_one_epoch(sess, step, data_generator, vars, writer):
    batch_per_epoch = vars['training_batch']
    iou_sum = 0
    iter = tqdm(range(batch_per_epoch)) if is_hvd_root else range(batch_per_epoch)
    for _ in iter:
        coors, features, num_list, bboxes = next(data_generator)
        iou, _, step, summary = \
            sess.run([vars['iou'],
                      vars['train_op'],
                      vars['global_step'],
                      vars['summary']],
                     feed_dict={vars['input_coors_p']: coors,
                                vars['input_features_p']: features,
                                vars['input_num_list_p']: num_list,
                                vars['input_bbox_p']: bboxes,
                                vars['is_training_p']: True})

        iou_sum += iou
        if is_hvd_root:
            writer.add_summary(summary, step)

    iou = iou_sum / batch_per_epoch

    if is_hvd_root:
        summary = tf.Summary()
        summary.value.add(tag='Overall IoU', simple_value=iou)
        writer.add_summary(summary, step)
        print("Training: Total IoU={:0.2f}".format(iou))
    return step


def valid_one_epoch(sess, data_generator, vars, writer):
    batch_per_epoch = vars['validation_batch']
    iou_sum = 0
    instance_count = 0
    iter = tqdm(range(batch_per_epoch)) if is_hvd_root else range(batch_per_epoch)

    for _, batch_id in enumerate(iter):
        coors, features, num_list, bboxes = next(data_generator)
        iou, step, summary = \
            sess.run([vars['iou'],
                      vars['global_step'],
                      vars['summary']],
                     feed_dict={vars['input_coors_p']: coors,
                                vars['input_features_p']: features,
                                vars['input_num_list_p']: num_list,
                                vars['input_bbox_p']: bboxes,
                                vars['is_training_p']: False})
        iou_sum += (iou * len(features))
        instance_count += len(features)
        # print(output, iou)

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
        for epoch in range(config.training_epochs):
            if is_hvd_root:
                print("Epoch: {}".format(epoch))
            step = train_one_epoch(mon_sess, step, train_generator, vars, training_writer)
            if epoch % config.valid_interval == 0:  # and EPOCH != 0:
                result = valid_one_epoch(mon_sess, valid_generator, vars, validation_writer)
                if is_hvd_root:
                    best_result = save_best_sess(mon_sess, best_result, result,
                                                 log_dir, saver, replace=False, log=is_hvd_root, inverse=False,
                                                 save_anyway=False)


if __name__ == '__main__':
    main()
