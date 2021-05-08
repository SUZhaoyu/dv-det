from __future__ import division

import os
from os.path import join
from shutil import copyfile

import numpy as np
import tensorflow as tf


def log_dir_setup(home, config_file):
    task_name = None
    dir_exist = True
    clean = False
    logdir = None
    while dir_exist:
        task_name = input("Input the task name: \n")
        if len(task_name) == 0:
            raise NameError("ERROR: task name cannot be empty.")
        logdir = join(home, 'checkpoints', task_name)
        dir_exist = os.path.isdir(logdir)
        if dir_exist:
            clean_tag = input("WARNING: log dir '{}' already exists, you want to clean it? [y/n]\n".format(logdir))
            if clean_tag == 'y':
                clean = True
                dir_exist = False
            elif clean_tag == 'n':
                print("INFO: continue with the existing dir '{}'".format(logdir))
                dir_exist = False
            else:
                raise SystemExit("ERROR: Only 'y' or 'n' is acceptable, program exit.")
    if clean:
        os.system('rm -rf {}'.format(logdir))
        print("WARNING: {} has been cleaned".format(logdir))
        os.mkdir(logdir)
    else:
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
    copyfile(config_file, join(logdir, config_file.split('/')[-1]))
    return logdir, task_name


def get_bn_decay(init_decay, current_step, decay_step, decay_rate, name='bn_decay'):
    if init_decay is not None:
        bn_momentum = tf.train.exponential_decay(
            init_decay,
            current_step,
            decay_step,
            decay_rate,
            staircase=True)
        bn_decay = tf.minimum(0.95, 1 - bn_momentum, name=name)
        tf.summary.scalar(name, bn_decay)
        return bn_decay
    else:
        return None


def get_weight_decay(init_decay, current_step, decay_step, decay_rate, limit=2e-5, name='weight_decay'):
    if init_decay is not None:
        wd_decay = tf.train.exponential_decay(
            init_decay,
            current_step,
            decay_step,
            decay_rate,
            staircase=True)
        wd_decay = tf.maximum(wd_decay, limit, name=name)
        tf.summary.scalar(name, wd_decay)
        return wd_decay
    else:
        return None


def get_learning_rate(init_lr, current_step, decay_step, decay_rate, hvd_size=1, lr_scale=False,
                      warm_up=False, name='learning_rate'):
    if not lr_scale:
        hvd_size = 1
    decay_learning_rate = tf.train.exponential_decay(
        init_lr,  # Base learning rate.
        current_step,  # Current number of step.
        decay_step,  # Decay step.
        decay_rate,  # Decay rate.
        staircase=True)
    if warm_up:
        warm_up_learning_rate = 0.9 * init_lr / decay_step * tf.cast(tf.identity(current_step),
                                                                             dtype=tf.float32) + 0.1 * init_lr
        # warm_up_learning_rate = tf.cast(warm_up_learning_rate, dtype=tf.float32)
        learning_rate = tf.cond(tf.less_equal(current_step, int(decay_step)),
                                lambda: warm_up_learning_rate,
                                lambda: decay_learning_rate)
    else:
        learning_rate = decay_learning_rate
    learning_rate = tf.maximum(learning_rate, 1e-7, name=name) * hvd_size  # CLIP THE LEARNING RATE!
    tf.summary.scalar(name, learning_rate)
    return learning_rate


def set_training_controls(config, lr, scale, decay_batch, step, lr_warm_up, hvd_size, prefix):
    lr = get_learning_rate(init_lr=lr,
                           current_step=step,
                           decay_step=decay_batch,
                           decay_rate=config.lr_decay,
                           name='{}_learning_rate'.format(prefix),
                           warm_up=lr_warm_up,
                           hvd_size=hvd_size,
                           lr_scale=scale)
    bn = get_bn_decay(init_decay=0.5,
                      current_step=step,
                      decay_step=decay_batch,
                      decay_rate=0.5,
                      name='{}_batch_decay'.format(prefix))
    wd = get_weight_decay(init_decay=config.weight_decay,
                          current_step=step,
                          decay_step=decay_batch,
                          decay_rate=0.8,
                          name='{}_weight_decay'.format(prefix))
    return lr, bn, wd



def get_iou_loss_weight(current_step, total_l1_step):
    return tf.cast(tf.greater_equal(current_step, total_l1_step), dtype=tf.float32)


# grad_check = tf.check_numerics(clipped_gradients)
# with tf.control_dependencies([grad_check]):
#   self.optimizer = opt.apply_gradients(zip(clipped_gradients, params))

def get_train_op(loss, learning_rate, opt='adam', var_keywords=None, global_step=None, use_hvd=False):
    if opt == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif opt == 'sgd':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True)
    else:
        raise NameError("Unsupported optimizer: {}".format(opt))
    if use_hvd:
        import horovod.tensorflow as hvd
        optimizer = hvd.DistributedOptimizer(optimizer, device_dense="/cpu:0")

    var_list = None
    if var_keywords is not None:
        var_list = []
        for keyword in var_keywords:
            for var in tf.trainable_variables():
                if keyword in var.name:
                    var_list.append(var)
        if len(var_list) == 0:
            print("WARNING: no variables matches with keywords: {}".format(var_keywords))
            var_list = None
    else:
        var_list = tf.trainable_variables()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        gradients = optimizer.compute_gradients(loss)
        # tf.summary.histogram("gradients", gradients)
        clipped_gradients = []
        for grad, var in gradients:
            if var in var_list:
                if grad is not None:
                    clipped_gradients.append((tf.clip_by_value(grad, -10., 10.), var))
                else:
                    clipped_gradients.append((grad, var))

        train_op = optimizer.apply_gradients(clipped_gradients, global_step=global_step)
    return train_op


def get_config(gpu=None):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = False
    config.log_device_placement = False
    if gpu is not None:
        config.gpu_options.visible_device_list = str(gpu)
    return config


def reset_metrics(scope='metrics'):
    op = tf.variables_initializer([var for var in tf.local_variables()
                                   if var.name.split('/')[0] == scope])
    return op


def get_horovod_session(sess):
    # https://github.com/tensorflow/tensorflow/issues/8425
    session = sess
    while type(session).__name__ != 'Session':
        session = session._sess
    return session


def save_sess(sess, label_idx, log_dir, saver, log=False):
    path = saver.save(get_horovod_session(sess), os.path.join(log_dir, 'test', 'model_{}'.format(label_idx)))
    if log:
        print("INFO: Model was saved to {}".format(path))


def save_best_sess(sess, best_acc, acc, log_dir, saver, replace=False, inverse=False, log=False, save_anyway=False):
    if save_anyway:
        path = saver.save(get_horovod_session(sess), os.path.join(log_dir, 'test', 'model_{}'.format(acc)))
        print("INFO: Model was saved to {}".format(path))
        return acc
    if not inverse:
        if acc > best_acc:
            if best_acc != 0 and replace:
                os.system('rm {}/{}*'.format(os.path.join(log_dir, 'test'), 'model_'))
            path = saver.save(get_horovod_session(sess), os.path.join(log_dir, 'test', 'model_{}'.format(acc)))
            best_acc = acc
            if log:
                print("INFO: Model was saved to {}".format(path))
    else:
        if acc < best_acc:
            if best_acc != np.inf and replace:
                os.system('rm {}/{}*'.format(os.path.join(log_dir, 'test'), 'model_'))
            path = saver.save(get_horovod_session(sess), os.path.join(log_dir, 'test', 'model_{}'.format(acc)))
            best_acc = acc
            if log:
                print("INFO: Model was saved to {}".format(path))
    return best_acc
