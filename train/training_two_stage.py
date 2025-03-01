from __future__ import division

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import horovod.tensorflow as hvd
import os
from tqdm import tqdm
from os.path import join, dirname
import sys
import argparse
from shutil import rmtree

HOME = join(dirname(os.getcwd()))
sys.path.append(HOME)

from models import rcnn_model as model
from train.configs import rcnn_config as config
from data.kitti_generator import Dataset
from train.train_utils import get_train_op, get_config, save_best_sess, set_training_controls

hvd.init()
is_hvd_root = hvd.rank() == 0

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

# if is_hvd_root:
#     copyfile(config.config_dir, join(log_dir, config.config_dir.split('/')[-1]))

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



is_stage1_training_p = tf.placeholder(dtype=tf.bool, shape=[], name="stage1_training")
is_stage2_training_p = tf.placeholder(dtype=tf.bool, shape=[], name="stage2_training")

stage1_step = tf.Variable(0, name='stage1_step')
stage2_step = tf.Variable(0, name='stage2_step')
stage1_lr, stage1_bn, stage1_wd = set_training_controls(config, decay_batch, stage1_step, hvd.size(), prefix='stage1')
stage2_lr, stage2_bn, stage2_wd = set_training_controls(config, decay_batch, stage2_step, hvd.size(), prefix='stage2')


coors, features, num_list, roi_coors, roi_attrs, roi_conf_logits, roi_num_list = \
    model.stage1_model(input_coors=input_coors_p,
                       input_features=input_features_p,
                       input_num_list=input_num_list_p,
                       is_training=is_stage1_training_p,
                       is_eval=False,
                       trainable=True,
                       mem_saving=True,
                       bn=stage1_bn)



bbox_attrs, bbox_conf_logits, bbox_num_list, bbox_idx = \
    model.stage2_model(coors=coors,
                       features=features,
                       num_list=num_list,
                       roi_attrs=roi_attrs,
                       roi_conf_logits=roi_conf_logits,
                       roi_num_list=roi_num_list,
                       is_training=is_stage2_training_p,
                       mem_saving=False,
                       trainable=True,
                       bn=stage2_bn)

stage1_loss, roi_ious, averaged_roi_iou = model.stage1_loss(roi_coors=roi_coors,
                                                            pred_roi_attrs=roi_attrs,
                                                            roi_conf_logits=roi_conf_logits,
                                                            roi_num_list=roi_num_list,
                                                            bbox_labels=input_bbox_p,
                                                            wd=stage1_wd)

stage2_loss, averaged_bbox_iou = model.stage2_loss(roi_attrs=roi_attrs,
                                                   pred_bbox_attrs=bbox_attrs,
                                                   bbox_conf_logits=bbox_conf_logits,
                                                   bbox_num_list=bbox_num_list,
                                                   bbox_labels=input_bbox_p,
                                                   bbox_idx=bbox_idx,
                                                   roi_ious=roi_ious,
                                                   wd=stage2_wd)

stage1_train_op = get_train_op(stage1_loss, stage1_lr, var_keywords=['stage1'], opt='adam', global_step=stage1_step, use_hvd=True)
stage2_train_op = get_train_op(stage2_loss, stage2_lr, var_keywords=['stage2'], opt='adam', global_step=stage2_step, use_hvd=True)

stage1_summary = tf.summary.merge_all(key='stage1')
stage2_summary = tf.summary.merge_all(key='stage2')
total_summary = tf.summary.merge_all()
hooks = [hvd.BroadcastGlobalVariablesHook(0)]
session_config = get_config(gpu=config.gpu_list[hvd.rank()])

training_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'))
validation_writer = tf.summary.FileWriter(os.path.join(log_dir, 'valid'))

saver = tf.train.Saver(max_to_keep=None)

vars = {'stage1_step': stage1_step,
        'stage2_step': stage2_step,
        'input_coors_p': input_coors_p,
        'input_features_p': input_features_p,
        'input_num_list_p': input_num_list_p,
        'input_bbox_p': input_bbox_p,
        'is_stage1_training_p': is_stage1_training_p,
        'is_stage2_training_p': is_stage2_training_p,
        'training_batch': training_batch,
        'validation_batch': validation_batch,
        'roi_iou': averaged_roi_iou,
        'bbox_iou': averaged_bbox_iou,
        'stage1_train_op': stage1_train_op,
        'stage2_train_op': stage2_train_op,
        'stage1_summary': total_summary,
        'stage2_summary': total_summary}


def train_one_epoch(sess, step, dataset_generator, vars, writer):
    batch_per_epoch = vars['training_batch']
    iou_sum = 0
    iter = tqdm(range(batch_per_epoch)) if is_hvd_root else range(batch_per_epoch)
    for _ in iter:
        is_stage1_training = step < config.stage1_training_epoch * batch_per_epoch
        is_stage2_training = not is_stage1_training
        train_op = vars['stage1_train_op'] if is_stage1_training else vars['stage2_train_op']
        output_iou = vars['roi_iou'] if is_stage1_training else vars['bbox_iou']
        tf_summary = vars['stage1_summary'] if is_stage1_training else vars['stage2_summary']
        coors, features, num_list, bboxes = next(dataset_generator)
        iou, _ = sess.run([output_iou, train_op],
                                    feed_dict={vars['input_coors_p']: coors,
                                               vars['input_features_p']: features,
                                               vars['input_num_list_p']: num_list,
                                               vars['input_bbox_p']: bboxes,
                                               vars['is_stage1_training_p']: is_stage1_training,
                                               vars['is_stage2_training_p']: is_stage2_training,})

        iou_sum += iou
        step += 1
        # if is_hvd_root:
        #     writer.add_summary(summary, step)

    iou = iou_sum / batch_per_epoch

    if is_hvd_root:
        summary = tf.Summary()
        summary.value.add(tag='Overall IoU', simple_value=iou)
        writer.add_summary(summary, step)
        print("Training: Total IoU={:0.2f}".format(iou))
    return step


def valid_one_epoch(sess, step, dataset_generator, vars, writer):
    batch_per_epoch = vars['validation_batch']
    iou_sum = 0
    instance_count = 0
    is_stage1_training = step < config.stage1_training_epoch * batch_per_epoch
    output_iou = vars['roi_iou'] if is_stage1_training else vars['bbox_iou']
    tf_summary = vars['stage1_summary'] if is_stage1_training else vars['stage2_summary']
    iter = tqdm(range(batch_per_epoch)) if is_hvd_root else range(batch_per_epoch)
    for _, batch_id in enumerate(iter):
        coors, features, num_list, bboxes = next(dataset_generator)
        iou = \
            sess.run(output_iou,
                     feed_dict={vars['input_coors_p']: coors,
                                vars['input_features_p']: features,
                                vars['input_num_list_p']: num_list,
                                vars['input_bbox_p']: bboxes,
                                vars['is_stage1_training_p']: False,
                                vars['is_stage2_training_p']: False})

        iou_sum += (iou * len(features))
        instance_count += len(features)
        # if is_hvd_root:
        #     writer.add_summary(summary, step + batch_id)

    iou = iou_sum / instance_count

    if is_hvd_root:
        summary = tf.Summary()
        summary.value.add(tag='Overall IoU', simple_value=iou)
        writer.add_summary(summary, step)
        print("Validation: Total IoU={:0.2f}".format(iou))
    return iou



def main():
    with tf.train.MonitoredTrainingSession(hooks=hooks, config=session_config) as mon_sess:
        # if is_hvd_root:
        #     saver.restore(mon_sess, '/home/tan/tony/dv-det/checkpoints/stage1/test/best_model_0.6361302239890648')
        train_generator = DatasetTrain.train_generator()
        valid_generator = DatasetValid.valid_generator()
        best_result = 0.
        step = 0
        for epoch in range(config.total_epoch):
            if is_hvd_root:
                print("Epoch: {}".format(epoch))
            step = train_one_epoch(mon_sess, step, train_generator, vars, training_writer)
            if epoch % config.valid_interval == 0:  # and EPOCH != 0:
                result = valid_one_epoch(mon_sess, step, valid_generator, vars, validation_writer)
                if is_hvd_root:
                    best_result = save_best_sess(mon_sess, best_result, result,
                                                 log_dir, saver, replace=False, log=is_hvd_root, inverse=False,
                                                 save_anyway=False)


if __name__ == '__main__':
    main()
