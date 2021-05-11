from __future__ import division

import tensorflow as tf
import numpy as np

from models.tf_ops.loader.sampling import grid_sampling, grid_sampling_thrust, voxel_sampling_idx_binary, \
    voxel_sampling_idx, voxel_sampling_feature
from models.tf_ops.loader.pooling import bev_projection
from models.utils.layers_wrapper import kernel_conv_wrapper, conv_1d_wrapper, dense_conv_wrapper, conv_3d_wrapper, conv_2d_wrapper


def point_conv(input_coors,
               input_features,
               input_num_list,
               voxel_idx,
               center_idx,
               layer_params,
               dimension_params,
               grid_buffer_size,
               output_pooling_size,
               scope,
               is_training,
               mem_saving,
               model_params,
               trainable=True,
               bn_decay=None,
               histogram=False,
               summary=False,
               second_last_layer=False,
               last_layer=False):

    if last_layer or second_last_layer:
        bn_decay = None
    # if bn_decay is not None:
    #     print("*****************************", scope)
    activation = model_params['activation'] if not last_layer else None
    # grid_sampling_method = grid_sampling_thrust if mem_saving else grid_sampling
    grid_sampling_method = grid_sampling
    voxel_sampling_idx_method = voxel_sampling_idx_binary if mem_saving else voxel_sampling_idx
    # voxel_sampling_method = voxel_sampling_binary if mem_saving else voxel_sampling

    if layer_params['subsample_res'] is not None:
        kernel_center_coors, center_num_list, center_idx = \
            grid_sampling_method(input_coors=input_coors,
                                 input_num_list=input_num_list,
                                 resolution=layer_params['subsample_res'],
                                 dimension=dimension_params['dimension'],
                                 offset=dimension_params['offset'])
    else:
        kernel_center_coors = input_coors
        center_num_list = input_num_list
        center_idx = center_idx

    if layer_params['kernel_res'] is not None:
        voxel_idx, features = voxel_sampling_idx_method(input_coors=input_coors,
                                                        input_features=input_features,
                                                        input_num_list=input_num_list,
                                                        center_coors=kernel_center_coors,
                                                        center_num_list=center_num_list,
                                                        resolution=layer_params['kernel_res'],
                                                        dimension=dimension_params['dimension'],
                                                        offset=dimension_params['offset'],
                                                        grid_buffer_size=grid_buffer_size,
                                                        output_pooling_size=output_pooling_size)
    else:
        voxel_idx = tf.gather(voxel_idx, center_idx, axis=0)
        features = input_features

    voxel_features = voxel_sampling_feature(input_features=features,
                                            output_idx=voxel_idx,
                                            padding=model_params['padding'])


    output_features = kernel_conv_wrapper(inputs=voxel_features,
                                          num_output_channels=layer_params['c_out'],
                                          scope=scope,
                                          trainable=trainable,
                                          use_xavier=model_params['xavier'],
                                          stddev=model_params['stddev'],
                                          activation=activation,
                                          bn_decay=bn_decay,
                                          is_training=is_training,
                                          histogram=histogram,
                                          summary=summary)

    return kernel_center_coors, output_features, center_num_list, voxel_idx, center_idx


def rpn_point_conv(input_coors,
                   input_features,
                   input_num_list,
                   center_coors,
                   center_num_list,
                   layer_params,
                   dimension_params,
                   grid_buffer_size,
                   output_pooling_size,
                   scope,
                   is_training,
                   mem_saving,
                   model_params,
                   trainable=True,
                   bn_decay=None,
                   histogram=False,
                   summary=False,
                   second_last_layer=False,
                   last_layer=False):


    activation = model_params['activation'] if not last_layer else None
    # grid_sampling_method = grid_sampling_thrust if mem_saving else grid_sampling
    grid_sampling_method = grid_sampling
    voxel_sampling_idx_method = voxel_sampling_idx_binary if mem_saving else voxel_sampling_idx


    voxel_idx, _, features = voxel_sampling_idx_method(input_coors=input_coors,
                                                       input_features=input_features,
                                                       input_num_list=input_num_list,
                                                       center_coors=center_coors,
                                                       center_num_list=center_num_list,
                                                       resolution=layer_params['kernel_res'],
                                                       dimension=dimension_params['dimension'],
                                                       offset=dimension_params['offset'],
                                                       grid_buffer_size=grid_buffer_size,
                                                       output_pooling_size=output_pooling_size)


    voxel_features = voxel_sampling_feature(input_features=features,
                                            output_idx=voxel_idx,
                                            padding=model_params['padding'])


    output_features = kernel_conv_wrapper(inputs=voxel_features,
                                          num_output_channels=layer_params['c_out'],
                                          scope=scope,
                                          trainable=trainable,
                                          use_xavier=model_params['xavier'],
                                          stddev=model_params['stddev'],
                                          activation=activation,
                                          bn_decay=bn_decay,
                                          is_training=is_training,
                                          histogram=histogram,
                                          summary=summary)

    return output_features


def point_conv_res(input_coors,
                   input_features,
                   input_num_list,
                   voxel_idx,
                   center_idx,
                   layer_params,
                   dimension_params,
                   grid_buffer_size,
                   output_pooling_size,
                   scope,
                   is_training,
                   mem_saving,
                   model_params,
                   trainable=True,
                   bn_decay=None,
                   histogram=False,
                   summary=False,
                   last_layer=False):
    input_channels = input_features.get_shape()[-1].value
    output_channels = layer_params['c_out']
    bn_decay = bn_decay if not last_layer else None
    activation = model_params['activation'] if not last_layer else None
    # grid_sampling_method = grid_sampling_thrust if mem_saving else grid_sampling
    grid_sampling_method = grid_sampling
    voxel_sampling_idx_method = voxel_sampling_idx_binary if mem_saving else voxel_sampling_idx

    if layer_params['subsample_res'] is not None:
        kernel_center_coors, center_num_list, center_idx = \
            grid_sampling_method(input_coors=input_coors,
                                 input_num_list=input_num_list,
                                 resolution=layer_params['subsample_res'],
                                 dimension=dimension_params['dimension'],
                                 offset=dimension_params['offset'])
    else:
        kernel_center_coors = input_coors
        center_num_list = input_num_list
        center_idx = center_idx

    res_features = tf.gather(input_features, center_idx, axis=0)
    if output_channels != input_channels:
        res_features = conv_1d_wrapper(inputs=res_features,
                                       num_output_channels=output_channels,
                                       scope=scope + '_res',
                                       use_xavier=model_params['xavier'],
                                       stddev=model_params['stddev'],
                                       activation=activation,
                                       bn_decay=bn_decay,
                                       is_training=is_training,
                                       trainable=trainable,
                                       histogram=histogram,
                                       summary=summary)

    # padding_features = tf.zeros(shape=[tf.shape(res_features)[0], output_channels - input_channels], dtype=tf.float32)
    # res_features = tf.concat([res_features, padding_features], axis=-1)

    # ================================== Conv 1x1x1 ===================================

    compress_features = conv_1d_wrapper(inputs=input_features,
                                        num_output_channels=input_channels//2,
                                        scope=scope+'_compress',
                                        use_xavier=model_params['xavier'],
                                        stddev=model_params['stddev'],
                                        activation=activation,
                                        bn_decay=bn_decay,
                                        is_training=is_training,
                                        trainable=trainable,
                                        histogram=histogram,
                                        summary=summary)

    # ================================== Conv 3x3x3 ==================================='

    if layer_params['kernel_res'] is not None:
        voxel_idx, features = voxel_sampling_idx_method(input_coors=input_coors,
                                                        input_features=compress_features,
                                                        input_num_list=input_num_list,
                                                        center_coors=kernel_center_coors,
                                                        center_num_list=center_num_list,
                                                        resolution=layer_params['kernel_res'],
                                                        dimension=dimension_params['dimension'],
                                                        offset=dimension_params['offset'],
                                                        grid_buffer_size=grid_buffer_size,
                                                        output_pooling_size=output_pooling_size)
    else:
        voxel_idx = tf.gather(voxel_idx, center_idx, axis=0)
        features = input_features

    voxel_features = voxel_sampling_feature(input_features=features,
                                            output_idx=voxel_idx,
                                            padding=model_params['padding'])

    output_features = kernel_conv_wrapper(inputs=voxel_features,
                                          num_output_channels=output_channels//2,
                                          scope=scope+'_voxel_conv',
                                          trainable=trainable,
                                          use_xavier=model_params['xavier'],
                                          stddev=model_params['stddev'],
                                          activation=activation,
                                          bn_decay=bn_decay,
                                          is_training=is_training,
                                          histogram=histogram,
                                          summary=summary)

    # ================================== Conv 1x1x1 ===================================

    decompress_features = conv_1d_wrapper(inputs=output_features,
                                          num_output_channels=output_channels,
                                          scope=scope+'_decompress',
                                          use_xavier=model_params['xavier'],
                                          stddev=model_params['stddev'],
                                          activation=None,
                                          bn_decay=bn_decay,
                                          is_training=is_training,
                                          trainable=trainable,
                                          histogram=histogram,
                                          summary=summary)

    # ================================== Res Addition ===================================

    ret_features = tf.add(decompress_features, res_features)
    ret_features = tf.nn.relu(ret_features)

    return kernel_center_coors, ret_features, center_num_list, voxel_idx, center_idx


def point_conv_concat(input_coors,
                      input_features,
                      concat_features,
                      input_num_list,
                      voxel_idx,
                      center_idx,
                      layer_params,
                      dimension_params,
                      grid_buffer_size,
                      output_pooling_size,
                      scope,
                      is_training,
                      mem_saving,
                      model_params,
                      trainable=True,
                      bn_decay=None,
                      histogram=False,
                      summary=False,
                      last_layer=False):
    bn_decay = bn_decay if not last_layer else None
    activation = model_params['activation'] if not last_layer else None
    # grid_sampling_method = grid_sampling_thrust if mem_saving else grid_sampling
    grid_sampling_method = grid_sampling
    voxel_sampling_idx_method = voxel_sampling_idx_binary if mem_saving else voxel_sampling_idx
    # voxel_sampling_method = voxel_sampling_binary if mem_saving else voxel_sampling

    if layer_params['subsample_res'] is not None:
        kernel_center_coors, center_num_list, center_idx = \
            grid_sampling_method(input_coors=input_coors,
                                 input_num_list=input_num_list,
                                 resolution=layer_params['subsample_res'],
                                 dimension=dimension_params['dimension'],
                                 offset=dimension_params['offset'])
        if concat_features is not None:
            concat_features = tf.gather(concat_features, center_idx, axis=0)
    else:
        kernel_center_coors = input_coors
        center_num_list = input_num_list
        center_idx = center_idx

    if layer_params['kernel_res'] is not None:
        voxel_idx, _, features = voxel_sampling_idx_method(input_coors=input_coors,
                                                           input_features=input_features,
                                                           input_num_list=input_num_list,
                                                           center_coors=kernel_center_coors,
                                                           center_num_list=center_num_list,
                                                           resolution=layer_params['kernel_res'],
                                                           dimension=dimension_params['dimension'],
                                                           offset=dimension_params['offset'],
                                                           grid_buffer_size=grid_buffer_size,
                                                           output_pooling_size=output_pooling_size,
                                                           with_rpn=False)
    else:
        if layer_params['subsample_res'] is not None:
            voxel_idx = tf.gather(voxel_idx, center_idx, axis=0)
            features = input_features
        else:
            voxel_idx = voxel_idx
            features = input_features


    voxel_features = voxel_sampling_feature(input_features=features,
                                            output_idx=voxel_idx,
                                            padding=model_params['padding'])

    output_features = kernel_conv_wrapper(inputs=voxel_features,
                                          num_output_channels=layer_params['c_out'],
                                          scope=scope,
                                          trainable=trainable,
                                          use_xavier=model_params['xavier'],
                                          stddev=model_params['stddev'],
                                          activation=activation,
                                          bn_decay=bn_decay,
                                          is_training=is_training,
                                          histogram=histogram,
                                          summary=summary)
    if concat_features is not None and layer_params['concat']:
        concat_features = tf.concat([concat_features, output_features], axis=1)

    return kernel_center_coors, output_features, center_num_list, voxel_idx, center_idx, concat_features


def point_conv_bev_concat(input_coors,
                          input_features,
                          concat_features,
                          input_num_list,
                          voxel_idx,
                          center_idx,
                          layer_params,
                          dimension_params,
                          grid_buffer_size,
                          output_pooling_size,
                          scope,
                          is_training,
                          mem_saving,
                          model_params,
                          trainable=True,
                          bn_decay=None,
                          histogram=False,
                          summary=False,
                          last_layer=False):
    bn_decay = bn_decay if not last_layer else None
    activation = model_params['activation'] if not last_layer else None
    # grid_sampling_method = grid_sampling_thrust if mem_saving else grid_sampling
    grid_sampling_method = grid_sampling
    voxel_sampling_idx_method = voxel_sampling_idx_binary if mem_saving else voxel_sampling_idx

    if layer_params['subsample_res'] is not None:
        kernel_center_coors, center_num_list, center_idx = \
            grid_sampling_method(input_coors=input_coors,
                                 input_num_list=input_num_list,
                                 resolution=layer_params['subsample_res'],
                                 dimension=dimension_params['dimension'],
                                 offset=dimension_params['offset'])
    else:
        kernel_center_coors = input_coors
        center_num_list = input_num_list
        center_idx = center_idx

    if layer_params['kernel_res'] is not None:
        voxel_idx, _, features = voxel_sampling_idx_method(input_coors=input_coors,
                                                           input_features=input_features,
                                                           input_num_list=input_num_list,
                                                           center_coors=kernel_center_coors,
                                                           center_num_list=center_num_list,
                                                           resolution=layer_params['kernel_res'],
                                                           dimension=dimension_params['dimension'],
                                                           offset=dimension_params['offset'],
                                                           grid_buffer_size=grid_buffer_size,
                                                           output_pooling_size=output_pooling_size,
                                                           with_rpn=False)
    else:
        if layer_params['subsample_res'] is not None:
            voxel_idx = tf.gather(voxel_idx, center_idx, axis=0)
            features = input_features
        else:
            voxel_idx = voxel_idx
            features = input_features


    voxel_features = voxel_sampling_feature(input_features=features,
                                            output_idx=voxel_idx,
                                            padding=model_params['padding'])

    output_features = kernel_conv_wrapper(inputs=voxel_features,
                                          num_output_channels=layer_params['c_out'],
                                          scope=scope,
                                          trainable=trainable,
                                          use_xavier=model_params['xavier'],
                                          stddev=model_params['stddev'],
                                          activation=activation,
                                          bn_decay=bn_decay,
                                          is_training=is_training,
                                          histogram=histogram,
                                          summary=summary)
    if concat_features is not None and layer_params['concat']:
        bev_img = bev_projection(input_coors=kernel_center_coors,
                                 input_features=output_features,
                                 input_num_list=center_num_list,
                                 offset=dimension_params['offset'],
                                 dimension=dimension_params['dimension'],
                                 resolution=layer_params['bev_res'])
        bev_img = conv_2d_wrapper(inputs=bev_img,
                                  num_output_channels=model_params['bev_channels'],
                                  output_shape=dimension_params['bev_size'],
                                  stride=layer_params['bev_stride'],
                                  scope=scope + '_bev',
                                  transposed=True)
        concat_features.append(bev_img)

    return kernel_center_coors, output_features, center_num_list, voxel_idx, center_idx, concat_features


def conv_3d(input_voxels,
            layer_params,
            scope,
            is_training,
            model_params,
            mem_saving,
            trainable=True,
            bn_decay=None,
            histogram=False,
            summary=False,
            last_layer=False):
    bn_decay = bn_decay if not last_layer else None
    activation = model_params['activation'] if not last_layer else None
    conv3d_method = conv_3d_wrapper if mem_saving else dense_conv_wrapper
    output_features = conv3d_method(inputs=input_voxels,
                                    num_output_channels=layer_params['c_out'],
                                    kernel_size=layer_params['kernel_size'],
                                    scope=scope,
                                    trainable=trainable,
                                    use_xavier=model_params['xavier'],
                                    stddev=model_params['stddev'],
                                    activation=activation,
                                    bn_decay=bn_decay,
                                    is_training=is_training,
                                    histogram=histogram,
                                    summary=summary)

    return output_features

def conv_3d_res(input_voxels,
                layer_params,
                scope,
                is_training,
                model_params,
                mem_saving,
                trainable=True,
                bn_decay=None,
                histogram=False,
                summary=False,
                last_layer=False):
    bn_decay = bn_decay if not last_layer else None
    activation = model_params['activation'] if not last_layer else None
    conv3d_method = conv_3d_wrapper if mem_saving else dense_conv_wrapper
    input_voxel_size = np.cbrt(input_voxels.get_shape()[1].value).astype(np.int32)
    input_channels = input_voxels.get_shape()[2].value
    output_channels = layer_params['c_out']
    print(input_channels, output_channels, input_voxel_size)

    input_features = tf.reshape(input_voxels, shape=[-1, input_channels])
    compress_features = conv_1d_wrapper(inputs=input_features,
                                        num_output_channels=input_channels // 4,
                                        scope=scope + '_compress',
                                        use_xavier=model_params['xavier'],
                                        stddev=model_params['stddev'],
                                        activation=activation,
                                        bn_decay=bn_decay,
                                        is_training=is_training,
                                        trainable=trainable,
                                        histogram=histogram,
                                        summary=summary)
    compress_features = tf.reshape(compress_features, shape=[-1, input_voxel_size * input_voxel_size * input_voxel_size, input_channels // 4])

    output_features = conv3d_method(inputs=compress_features,
                                    num_output_channels=output_channels // 4,
                                    kernel_size=layer_params['kernel_size'],
                                    scope=scope + '_3x3x3_conv',
                                    trainable=trainable,
                                    use_xavier=model_params['xavier'],
                                    stddev=model_params['stddev'],
                                    activation=activation,
                                    bn_decay=bn_decay,
                                    is_training=is_training,
                                    histogram=histogram,
                                    summary=summary)
    output_voxel_size = np.cbrt(output_features.get_shape()[1].value).astype(np.int32)

    output_features = tf.reshape(output_features, shape=[-1, output_channels // 4])
    decompress_features = conv_1d_wrapper(inputs=output_features,
                                          num_output_channels=output_channels,
                                          scope=scope + '_decompress',
                                          use_xavier=model_params['xavier'],
                                          stddev=model_params['stddev'],
                                          activation=activation,
                                          bn_decay=bn_decay,
                                          is_training=is_training,
                                          trainable=trainable,
                                          histogram=histogram,
                                          summary=summary)
    decompress_features = tf.reshape(decompress_features, shape=[-1, output_voxel_size * output_voxel_size * output_voxel_size, output_channels])

    return decompress_features


def conv_1d(input_points,
            num_output_channels,
            drop_rate,
            model_params,
            scope,
            is_training,
            trainable=True,
            bn_decay=None,
            histogram=False,
            summary=False,
            second_last_layer=False,
            last_layer=False):
    inputs = tf.nn.dropout(input_points, rate=drop_rate)
    activation = model_params['activation'] if not last_layer else None
    if last_layer or second_last_layer:
        bn_decay = None
    output_points = conv_1d_wrapper(inputs=inputs,
                                    num_output_channels=num_output_channels,
                                    scope=scope,
                                    use_xavier=model_params['xavier'],
                                    stddev=model_params['stddev'],
                                    activation=activation,
                                    bn_decay=bn_decay,
                                    is_training=is_training,
                                    trainable=trainable,
                                    histogram=histogram,
                                    summary=summary)
    return output_points
