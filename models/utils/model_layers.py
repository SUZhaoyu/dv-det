from __future__ import division

import tensorflow as tf

from models.tf_ops.custom_ops import grid_sampling, voxel_sampling, roi_pooling
from models.utils.ops_wrapper import kernel_conv_wrapper, fully_connected_wrapper, dense_conv_wrapper


def point_conv(input_coors,
               input_features,
               input_num_list,
               layer_params,
               scope,
               is_training,
               model_params,
               bn_decay=None,
               histogram=False,
               summary=False,
               last_layer=False):
    bn_decay = bn_decay if not last_layer else None
    activation = model_params['activation'] if not last_layer else None
    if layer_params['subsample_res'] is not None:
        kernel_center_coors, center_num_list = grid_sampling(input_coors=input_coors,
                                                             input_num_list=input_num_list,
                                                             resolution=layer_params['subsample_res'],
                                                             dimension=model_params['dimension'],
                                                             offset=model_params['offset'])
    else:
        kernel_center_coors = input_coors
        center_num_list = input_num_list
    voxels = voxel_sampling(input_coors=input_coors,
                            input_features=input_features,
                            input_num_list=input_num_list,
                            center_coors=kernel_center_coors,
                            center_num_list=center_num_list,
                            resolution=layer_params['kernel_res'],
                            padding=layer_params['padding'],
                            dimension=model_params['dimension'],
                            offset=model_params['offset'])

    output_features = kernel_conv_wrapper(inputs=voxels,
                                          num_output_channels=layer_params['c_out'],
                                          scope=scope,
                                          use_xavier=model_params['xavier'],
                                          stddev=model_params['stddev'],
                                          activation=activation,
                                          bn_decay=bn_decay,
                                          is_training=is_training,
                                          histogram=histogram,
                                          summary=summary)
    return kernel_center_coors, output_features, center_num_list


def conv_3d(input_voxels,
            layer_params,
            scope,
            is_training,
            model_params,
            bn_decay=None,
            histogram=False,
            summary=False,
            last_layer=False):
    bn_decay = bn_decay if not last_layer else None
    activation = model_params['activation'] if not last_layer else None

    output_features = dense_conv_wrapper(inputs=input_voxels,
                                         num_output_channels=layer_params['c_out'],
                                         kernel_size=layer_params['kernel_size'],
                                         scope=scope,
                                         use_xavier=model_params['xavier'],
                                         stddev=model_params['stddev'],
                                         activation=activation,
                                         bn_decay=bn_decay,
                                         is_training=is_training,
                                         histogram=histogram,
                                         summary=summary)

    return output_features


def fully_connected(input_points,
                    num_output_channels,
                    drop_rate,
                    model_params,
                    scope,
                    is_training,
                    bn_decay=None,
                    histogram=False,
                    summary=False,
                    last_layer=False):
    inputs = tf.nn.dropout(input_points, rate=drop_rate)
    activation = model_params['activation'] if not last_layer else None
    bn_decay = bn_decay if not last_layer else None
    output_points = fully_connected_wrapper(inputs=inputs,
                                            num_output_channels=num_output_channels,
                                            scope=scope,
                                            use_xavier=model_params['xavier'],
                                            stddev=model_params['stddev'],
                                            activation=activation,
                                            bn_decay=bn_decay,
                                            is_training=is_training,
                                            histogram=histogram,
                                            summary=summary)
    return output_points
