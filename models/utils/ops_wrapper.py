import tensorflow as tf

from models.utils.var_utils import _variable_with_l2_loss


# from models.tf_ops.custom_ops import kernel_conv

def batch_norm_template(inputs, is_training, bn_decay, name):
    '''
    Batch Norm for voxel conv operation

    :param inputs: Tensor, 3D [batch, nkernel, channel], coming from voxel conv
    :param is_training: boolean tf.Variable, true indicated training phase
    :param bn_decay: float or float tensor variable, controling moving average weight
    :param scope: string, variable scope

    :return: batch-normalized maps
    '''
    return tf.layers.batch_normalization(inputs=inputs,
                                         axis=-1,
                                         momentum=bn_decay,
                                         training=is_training,
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
                        histogram=False,
                        summary=False):
    if scope == 'default':
        print("WARNING: scope name was not given and has been assigned as 'default' ")
    l2_loss_collection = "l2"
    with tf.variable_scope(scope):
        num_input_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_size ** 3 * num_input_channels, num_output_channels]
        kernel = _variable_with_l2_loss(name='kernel',
                                        shape=kernel_shape,
                                        use_xavier=use_xavier,
                                        stddev=stddev,
                                        l2_loss_collection=l2_loss_collection)
        biases = _variable_with_l2_loss(name='biases',
                                        shape=[num_output_channels],
                                        initializer=tf.constant_initializer(0.0),
                                        with_l2_loss=False,
                                        l2_loss_collection=None)
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
                                          name='bn')
        if activation is not None:
            activation_fn_dict = {'relu': tf.nn.relu,
                                  'elu': tf.nn.elu,
                                  'leaky_relu': tf.nn.leaky_relu}
            activation_fn = activation_fn_dict[activation]
            outputs = activation_fn(outputs)
        return outputs


def fully_connected_wrapper(inputs,
                            num_output_channels,
                            scope='default',
                            use_xavier=True,
                            stddev=1e-3,
                            activation='relu',
                            bn_decay=None,
                            is_training=True,
                            histogram=False,
                            summary=False):
    if scope == 'default':
        print("WARNING: scope name was not given and has been assigned as 'default' ")
    l2_loss_collection = "l2"
    with tf.variable_scope(scope):
        num_input_channels = inputs.get_shape()[-1].value
        weight_shape = [num_input_channels, num_output_channels]
        weight = _variable_with_l2_loss(name='weight',
                                        shape=weight_shape,
                                        use_xavier=use_xavier,
                                        stddev=stddev,
                                        l2_loss_collection=l2_loss_collection)
        biases = _variable_with_l2_loss(name="biases",
                                        shape=[num_output_channels],
                                        initializer=tf.constant_initializer(0.0),
                                        with_l2_loss=False,
                                        l2_loss_collection=None)
        if histogram:
            tf.summary.histogram('weight', weight)
        if summary:
            tf.summary.scalar('weight_L2', tf.nn.l2_loss(weight))
        outputs = tf.matmul(inputs, weight)
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
