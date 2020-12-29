import tensorflow as tf

def focal_loss(label, pred, alpha=0.25, gamma=2):
    part_a = -alpha * (1 - pred) ** gamma * tf.log(pred) * label
    part_b = -(1 - alpha) * pred ** gamma * tf.log(1 - pred) * (1 - label)
    return part_a + part_b


def smooth_l1_loss(predictions, labels, delta=1.0, use_sin=False):
    if use_sin:
        residual = tf.abs(tf.sin(predictions - labels))
    else:
        residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def get_masked_average(input, mask):
    return tf.math.divide_no_nan(tf.reduce_sum(input * mask), tf.reduce_sum(mask))