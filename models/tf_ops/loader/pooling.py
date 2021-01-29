import os
from os.path import join

import tensorflow as tf
from tensorflow.python.framework import ops

CWD = os.path.dirname(os.path.abspath(__file__))

# =============================================Roi Pooling===============================================

roi_pooling_exe = tf.load_op_library(join(CWD, '../build', 'roi_pooling.so'))
def roi_pooling(input_coors, input_features, roi_attrs, input_num_list, roi_num_list,
                voxel_size=5, padding_value=0., pooling_size=5):
    output_features, _ = roi_pooling_exe.roi_pooling_op(input_coors=input_coors,
                                                        input_features=input_features,
                                                        roi_attrs=roi_attrs,
                                                        input_num_list=input_num_list,
                                                        roi_num_list=roi_num_list,
                                                        voxel_size=voxel_size,
                                                        padding_value=padding_value,
                                                        pooling_size=pooling_size)
    return output_features

@ops.RegisterGradient("RoiPoolingOp")
def roi_pooling_grad(op, grad, _):
    input_features = op.inputs[1]
    output_idx = op.outputs[1]
    input_features_grad = roi_pooling_exe.roi_pooling_grad_op(input_features=input_features,
                                                              output_idx=output_idx,
                                                              output_features_grad=grad)
    return [None, input_features_grad, None, None, None]

# =============================================La Roi Pooling===============================================

la_roi_pooling_exe = tf.load_op_library(join(CWD, '../build', 'la_roi_pooling.so'))
def la_roi_pooling(input_coors, input_features, roi_attrs, input_num_list, roi_num_list,
                voxel_size=5, padding_value=0., pooling_size=5):
    output_features, _, _ = la_roi_pooling_exe.la_roi_pooling_op(input_coors=input_coors,
                                                                 input_features=input_features,
                                                                 roi_attrs=roi_attrs,
                                                                 input_num_list=input_num_list,
                                                                 roi_num_list=roi_num_list,
                                                                 voxel_size=voxel_size,
                                                                 padding_value=padding_value,
                                                                 pooling_size=pooling_size)
    return output_features

@ops.RegisterGradient("LaRoiPoolingOp")
def la_roi_pooling_grad(op, grad, _, __):
    input_features = op.inputs[1]
    output_idx = op.outputs[1]
    output_weight = op.outputs[2]
    input_features_grad = la_roi_pooling_exe.la_roi_pooling_grad_op(input_features=input_features,
                                                                    output_idx=output_idx,
                                                                    output_weight=output_weight,
                                                                    output_features_grad=grad)
    return [None, input_features_grad, None, None, None]

# =============================================La Roi Pooling Fast===============================================

la_roi_pooling_fast_exe = tf.load_op_library(join(CWD, '../build', 'la_roi_pooling_fast.so'))
def la_roi_pooling_fast(input_coors, input_features, roi_attrs, input_num_list, roi_num_list,
                        dimension, offset, grid_buffer_resolution,
                        grid_buffer_size=3, voxel_size=5, padding_value=0., pooling_size=5):
    output_features, _, _ = la_roi_pooling_fast_exe.la_roi_pooling_fast_op(input_coors=input_coors + offset,
                                                                           input_features=input_features,
                                                                           roi_attrs=roi_attrs,
                                                                           input_num_list=input_num_list,
                                                                           roi_num_list=roi_num_list,
                                                                           voxel_size=voxel_size,
                                                                           padding_value=padding_value,
                                                                           pooling_size=pooling_size,
                                                                           dimension=dimension,
                                                                           offset=offset,
                                                                           grid_buffer_resolution=grid_buffer_resolution,
                                                                           grid_buffer_size=grid_buffer_size)
    return output_features
ops.NoGradient("LaRoiPoolingFastOp")


