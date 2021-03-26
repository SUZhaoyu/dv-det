import numpy as np
import tensorflow as tf

from models.tf_ops.custom_ops import roi_pooling, voxel_sampling, grid_sampling
from models.utils.layers_wrapper import dense_conv_wrapper, kernel_conv_wrapper, conv_1d_wrapper

def point_conv(input_coors, input_features, input_num_list, c_out, resolution, scope):
    coors, num_list = grid_sampling(input_coors=input_coors,
                                    input_num_list=input_num_list,
                                    resolution=resolution)

    voxels = voxel_sampling(input_coors=input_coors,
                            input_features=input_features,
                            input_num_list=input_num_list,
                            center_coors=coors,
                            center_num_list=num_list,
                            resolution=resolution,
                            padding=-1)

    features = kernel_conv_wrapper(inputs=voxels,
                                   num_output_channels=c_out,
                                   scope=scope,
                                   bn_decay=1.)

    return coors, features, num_list


def conv_3d(input_voxels, c_out, scope):
    output_features = dense_conv_wrapper(inputs=input_voxels,
                                         num_output_channels=c_out,
                                         kernel_size=3,
                                         scope=scope,
                                         bn_decay=1.)
    return output_features


def fully_connected(input_points, c_out, scope):
    output_points = conv_1d_wrapper(inputs=input_points,
                                    num_output_channels=c_out,
                                    scope=scope,
                                    bn_decay=1.)
    return output_points
