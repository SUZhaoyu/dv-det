import logging

import numpy as np
import tensorflow as tf

from models.tf_ops.loader.bbox_utils import roi_logits_to_attrs, bbox_logits_to_attrs
from models.tf_ops.loader.others import voxel2col
from models.utils.iou_utils import roi_logits_to_attrs_tf, bbox_logits_to_attrs_tf
from models.utils.ops_utils import _variable_with_l2_loss


def batch_norm_template(inputs, is_training, bn_decay, name, trainable=True):
    '''
    Batch Norm for voxel conv operation

    :param inputs: Tensor, 3D [batch, nkernel, channel], coming from voxel conv
    :param is_training: boolean tf.Variable, true indicated training phase
    :param bn_decay: float or float tensor variable, controling moving average weight
    :param scope: string, variable scope

    :return: batch-normalized maps
    '''
    # from horovod.tensorflow.sync_batch_norm import SyncBatchNormalization
    # hvd_sync_bn = SyncBatchNormalization(axis=-1,
    #                                      momentum=bn_decay,
    #                                      trainable=trainable,
    #                                      name=name)
    # ret = hvd_sync_bn(inputs, training=is_training)  # 调用后updates属性才会有内容。
    # # print(hvd_sync_bn.updates)
    # for op in hvd_sync_bn.updates:
    #     tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, op)
    #
    # return ret

    return tf.layers.batch_normalization(inputs=inputs,
                                         axis=-1,
                                         momentum=bn_decay,
                                         training=is_training,
                                         trainable=trainable,
                                         name=name)


def kernel_conv_wrapper(inputs,
                        num_output_channels,
                        kernel_size=3,
                        scope='default',
                        use_xavier=True,
                        stddev=1e-3,
                        activation='relu',
                        bn_decay=None,
                        is_training=True,
                        trainable=True,
                        histogram=False,
                        summary=False):
    if scope == 'default':
        logging.warning("Scope name was not given and has been assigned as 'default'. ")
        l2_loss_collection = 'default'
    else:
        l2_loss_collection = scope.split('_')[0] + "_l2"

    with tf.variable_scope(scope):
        num_input_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_size ** 3 * num_input_channels, num_output_channels]
        kernel = _variable_with_l2_loss(name='kernel',
                                        shape=kernel_shape,
                                        use_xavier=use_xavier,
                                        stddev=stddev,
                                        l2_loss_collection=l2_loss_collection,
                                        trainable=trainable)
        biases = _variable_with_l2_loss(name='biases',
                                        shape=[num_output_channels],
                                        initializer=tf.constant_initializer(0.0),
                                        with_l2_loss=False,
                                        l2_loss_collection=None,
                                        trainable=trainable)
        if histogram:
            tf.summary.histogram('kernel', kernel)
        if summary:
            tf.summary.scalar('kernel_l2', tf.nn.l2_loss(kernel))

        inputs = tf.reshape(inputs, shape=[-1, kernel_size ** 3 * num_input_channels])
        outputs = tf.matmul(inputs, kernel)
        # outputs = kernel_conv(input_voxels=inputs,
        #                       filter=kernel)

        if bn_decay is None:
            outputs = tf.nn.bias_add(outputs, biases)
        else:
            outputs = batch_norm_template(inputs=outputs,
                                          is_training=is_training,
                                          bn_decay=bn_decay,
                                          name='bn',
                                          trainable=trainable)
        if activation is not None:
            activation_fn_dict = {'relu': tf.nn.relu,
                                  'elu': tf.nn.elu,
                                  'leaky_relu': tf.nn.leaky_relu}
            activation_fn = activation_fn_dict[activation]
            outputs = activation_fn(outputs)
    return outputs

def dense_conv_wrapper(inputs,
                       num_output_channels,
                       kernel_size=3,
                       scope='default',
                       use_xavier=True,
                       stddev=1e-3,
                       activation='relu',
                       bn_decay=None,
                       is_training=True,
                       trainable=True,
                       histogram=False,
                       summary=False):
    if scope == 'default':
        logging.warning("Scope name was not given and has been assigned as 'default'. ")
        l2_loss_collection = 'default'
    else:
        l2_loss_collection = scope.split('_')[0] + "_l2"
    with tf.variable_scope(scope):
        num_input_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_size, kernel_size, kernel_size, num_input_channels, num_output_channels]
        kernel = _variable_with_l2_loss(name='weight',
                                        shape=kernel_shape,
                                        use_xavier=use_xavier,
                                        stddev=stddev,
                                        l2_loss_collection=l2_loss_collection,
                                        trainable=trainable)
        biases = _variable_with_l2_loss(name='biases',
                                        shape=[num_output_channels],
                                        initializer=tf.constant_initializer(0.0),
                                        with_l2_loss=False,
                                        l2_loss_collection=None,
                                        trainable=trainable)
        if histogram:
            tf.summary.histogram('kernel', kernel)
        if summary:
            tf.summary.scalar('kernel_L2', tf.nn.l2_loss(kernel))

        voxels_to_conv = voxel2col(input_voxels=inputs, kernel_size=kernel_size)
        kernel = tf.reshape(kernel, shape=[-1, num_output_channels])
        outputs = tf.matmul(voxels_to_conv, kernel)

        if bn_decay is None:
            outputs = tf.nn.bias_add(outputs, biases)
        else:
            outputs = batch_norm_template(inputs=outputs,
                                          is_training=is_training,
                                          bn_decay=bn_decay,
                                          name='bn',
                                          trainable=trainable)
        if activation is not None:
            activation_fn_dict = {'relu': tf.nn.relu,
                                  'elu': tf.nn.elu,
                                  'leaky_relu': tf.nn.leaky_relu}
            activation_fn = activation_fn_dict[activation]
            outputs = activation_fn(outputs)

    return outputs


def conv_1d_wrapper(inputs,
                    num_output_channels,
                    scope='default',
                    use_xavier=True,
                    stddev=1e-3,
                    activation='relu',
                    bn_decay=None,
                    is_training=True,
                    trainable=True,
                    histogram=False,
                    summary=False):
    if scope == 'default':
        logging.warning("Scope name was not given and has been assigned as 'default'. ")
        l2_loss_collection = 'default'
    else:
        l2_loss_collection = scope.split('_')[0] + "_l2"
    with tf.variable_scope(scope):
        num_input_channels = inputs.get_shape()[-1].value
        weight_shape = [num_input_channels, num_output_channels]
        weight = _variable_with_l2_loss(name='weight',
                                        shape=weight_shape,
                                        use_xavier=use_xavier,
                                        stddev=stddev,
                                        l2_loss_collection=l2_loss_collection,
                                        trainable=trainable)
        biases = _variable_with_l2_loss(name="biases",
                                        shape=[num_output_channels],
                                        initializer=tf.constant_initializer(0.0),
                                        with_l2_loss=False,
                                        l2_loss_collection=None,
                                        trainable=trainable)
        if histogram:
            tf.summary.histogram('weight', weight)
        if summary:
            tf.summary.scalar('weight_l2', tf.nn.l2_loss(weight))
        outputs = tf.matmul(inputs, weight)
        if bn_decay is None:
            outputs = tf.nn.bias_add(outputs, biases)
        else:
            outputs = batch_norm_template(inputs=outputs,
                                          is_training=is_training,
                                          bn_decay=bn_decay,
                                          name='bn',
                                          trainable=trainable)
        if activation is not None:
            activation_fn_dict = {'relu': tf.nn.relu,
                                  'elu': tf.nn.elu,
                                  'leaky_relu': tf.nn.leaky_relu}
            activation_fn = activation_fn_dict[activation]
            outputs = activation_fn(outputs)
    return outputs

def conv_3d_wrapper(inputs,
                    num_output_channels,
                    kernel_size=3,
                    scope='default',
                    use_xavier=True,
                    stddev=1e-3,
                    activation='relu',
                    bn_decay=None,
                    is_training=True,
                    trainable=True,
                    histogram=False,
                    summary=False):
    if scope == 'default':
        logging.warning("Scope name was not given and has been assigned as 'default'. ")
        l2_loss_collection = 'default'
    else:
        l2_loss_collection = scope.split('_')[0] + "_l2"
    with tf.variable_scope(scope):
        kernel_num = tf.shape(inputs)[0]
        num_input_channels = inputs.get_shape()[-1].value
        input_size = np.cbrt(inputs.get_shape()[-2].value).astype(np.int32)
        reshaped_inputs = tf.reshape(inputs, shape=[-1, input_size, input_size, input_size, num_input_channels])
        kernel_shape = [kernel_size, kernel_size, kernel_size, num_input_channels, num_output_channels]
        kernel = _variable_with_l2_loss(name='weight',
                                        shape=kernel_shape,
                                        use_xavier=use_xavier,
                                        stddev=stddev,
                                        l2_loss_collection=l2_loss_collection,
                                        trainable=trainable)
        biases = _variable_with_l2_loss(name='biases',
                                        shape=[num_output_channels],
                                        initializer=tf.constant_initializer(0.0),
                                        with_l2_loss=False,
                                        l2_loss_collection=None,
                                        trainable=trainable)
        if histogram:
            tf.summary.histogram('kernel', kernel)
        if summary:
            tf.summary.scalar('kernel_l2', tf.nn.l2_loss(kernel))

        # outputs = tf.nn.conv3d(input=reshaped_inputs,
        #                        filter=kernel,
        #                        strides=[1, 1, 1, 1, 1],
        #                        padding='VALID')

        outputs = tf.cond(tf.greater(kernel_num, 0),
                          lambda: tf.nn.conv3d(input=reshaped_inputs,
                                               filter=kernel,
                                               strides=[1, 1, 1, 1, 1],
                                               padding='VALID'),
                          lambda: tf.random.uniform(shape=[0, input_size-2, input_size-2, input_size-2, num_output_channels]))

        if bn_decay is None:
            outputs = tf.nn.bias_add(outputs, biases)
        else:
            outputs = batch_norm_template(inputs=outputs,
                                          is_training=is_training,
                                          bn_decay=bn_decay,
                                          name='bn')
        if activation is not None:
            activation_fn_dict = {'relu': tf.nn.relu,
                                  'elu': tf.nn.elu,
                                  'leaky_relu': tf.nn.leaky_relu}
            activation_fn = activation_fn_dict[activation]
            outputs = activation_fn(outputs)
        output_grid_num = int(outputs.get_shape()[1].value * outputs.get_shape()[2].value * outputs.get_shape()[3].value)
        outputs = tf.reshape(outputs, shape=[-1, output_grid_num, num_output_channels])

    return outputs


def conv_2d_wrapper(inputs,
                    num_output_channels,
                    stride,
                    output_shape=None,
                    kernel_size=3,
                    transposed=False,
                    scope='default',
                    use_xavier=True,
                    stddev=1e-3,
                    activation='relu',
                    bn_decay=None,
                    is_training=True,
                    trainable=True,
                    histogram=False,
                    summary=False):
    if scope == 'default':
        logging.warning("Scope name was not given and has been assigned as 'default'. ")
        l2_loss_collection = 'default'
    else:
        l2_loss_collection = scope.split('_')[0] + "_l2"

    with tf.variable_scope(scope):
        num_input_channels = inputs.get_shape()[-1].value
        if not transposed:
            kernel_shape = [kernel_size, kernel_size, num_input_channels, num_output_channels]
        else:
            kernel_shape = [kernel_size, kernel_size, num_output_channels, num_input_channels]
        kernel = _variable_with_l2_loss(name='weight',
                                        shape=kernel_shape,
                                        use_xavier=use_xavier,
                                        stddev=stddev,
                                        l2_loss_collection=l2_loss_collection,
                                        trainable=trainable)

        biases = _variable_with_l2_loss(name='biases',
                                        shape=[num_output_channels],
                                        initializer=tf.constant_initializer(0.0),
                                        with_l2_loss=False,
                                        l2_loss_collection=None,
                                        trainable=trainable)

        if histogram:
            tf.summary.histogram('kernel', kernel)
        if summary:
            tf.summary.scalar('kernel_l2', tf.nn.l2_loss(kernel))

        if not transposed:
            outputs = tf.nn.conv2d(input=inputs,
                                   filter=kernel,
                                   strides=stride,
                                   padding="SAME")
        else:

            outputs = tf.nn.conv2d_transpose(input=inputs,
                                             filter=kernel,
                                             output_shape=output_shape,
                                             strides=stride,
                                             padding="SAME")
            # outputs = tf.reshape(outputs, output_shape)

        if bn_decay is None:
            outputs = tf.nn.bias_add(outputs, biases)
        else:
            outputs = batch_norm_template(inputs=outputs,
                                          is_training=is_training,
                                          bn_decay=bn_decay,
                                          name='bn')
        if activation is not None:
            activation_fn_dict = {'relu': tf.nn.relu,
                                  'elu': tf.nn.elu,
                                  'leaky_relu': tf.nn.leaky_relu}
            activation_fn = activation_fn_dict[activation]
            outputs = activation_fn(outputs)

    return outputs


def get_roi_attrs(input_logits, base_coors, anchor_size, is_eval=False):
    method = roi_logits_to_attrs if is_eval else roi_logits_to_attrs_tf
    roi_attrs = method(input_logits=input_logits,
                       base_coors=base_coors,
                       anchor_size=anchor_size)
    return roi_attrs


def get_bbox_attrs(input_logits, input_roi_attrs, is_eval=False):
    method = bbox_logits_to_attrs if is_eval else bbox_logits_to_attrs_tf
    bbox_attrs = method(input_logits=input_logits,
                        input_roi_attrs=input_roi_attrs)
    return bbox_attrs