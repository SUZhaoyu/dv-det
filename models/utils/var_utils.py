import tensorflow as tf


def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device("/cpu:0"):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=True)
    return var


def _variable_on_gpu(name, shape, initializer, use_fp16=False, gpu_id=None):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    if gpu_id is not None:
        with tf.device("/gpu:" + str(gpu_id)):
            dtype = tf.float16 if use_fp16 else tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=True)
    else:
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=True)
    return var


def _variable_with_l2_loss(name,
                           shape,
                           stddev=None,
                           initializer=None,
                           use_xavier=True,
                           with_l2_loss=True,
                           l2_loss_collection=None):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      use_xavier: bool, whether to use xavier initializer
    Returns:
      Variable Tensor
    Addition:
        Truncated normal initializer:
                These values are similar to values from a `random_normal_initializer`
                except that values more than two standard deviations from the mean
                are discarded and re-drawn. This is the recommended initializer for
                neural network weights and filters.
        Xavier Glorot and Yoshua Bengio (2010):
                 [Understanding the difficulty of training deep feedforward neural
                 networks. International conference on artificial intelligence and
                 statistics.](
                 http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
        This initializer is designed to keep the scale of the gradients roughly the
        same in all layers. In uniform distribution this ends up being the range:
        `x = sqrt(6. / (in + out)); [-x, x]` and for normal distribution a standard
        deviation of `sqrt(2. / (in + out))` is used
    """
    if initializer is None:
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=True)
    if with_l2_loss:
        if l2_loss_collection is not None:
            tf.add_to_collection(l2_loss_collection, tf.nn.l2_loss(var))
        else:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(var))
    return var
