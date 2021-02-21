/* Voxel Sampling Operation
 * with unstructured number of input points for each mini batch
 * Created by Zhaoyu SU
 * All Rights Reserved. Sep., 2019.
 */
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <climits>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("VoxelSamplingFeatureOp")
    .Input("input_features: float32")
    .Input("output_idx: int32")
    .Output("output_features: float32") // [center_coors.shape[0], kernel_size ** 3, channels]
    .Attr("padding_value: float")
    .SetShapeFn([](InferenceContext* c){
        ShapeHandle input_features_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_features_shape));
        ShapeHandle output_idx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &output_idx_shape));

        int kernel_size = 3;

        DimensionHandle channels = c->Dim(input_features_shape, 1);
        DimensionHandle center_num = c->Dim(output_idx_shape, 0);

        // The output shape during the shape inference stage is pseudo.
        ShapeHandle output_feature_shape = c->MakeShape({center_num, kernel_size * kernel_size * kernel_size, channels});

        c->set_output(0, output_feature_shape); // output_features

        return Status::OK();

    }); // InferenceContext


void voxel_sampling_feature_gpu_launcher(int center_num, int kernel_num, int channels, float padding,
                                         int output_pooling_size,
                                         const float* input_features,
                                         const int* output_idx,
                                         float* output_features);

class VoxelSamplingFeatureOp: public OpKernel {
public:
    explicit VoxelSamplingFeatureOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("padding_value", &padding_value));
    }
    void Compute(OpKernelContext* context) override {

        const Tensor& input_features = context->input(0);
        auto input_features_ptr = input_features.template flat<float>().data();
        OP_REQUIRES(context, input_features.dims()==2 && input_features.dim_size(1) > 0,
                    errors::InvalidArgument("VoxelSamplingFeatureOp expects input_features in shape: [point_nums, channels(>0)]."));

        const Tensor& output_idx = context->input(1);
        auto output_idx_ptr = output_idx.template flat<int>().data();
        OP_REQUIRES(context, output_idx.dims()==3 && output_idx.dim_size(1)==27,
                    errors::InvalidArgument("VoxelSamplingFeatureOp expects output_idx in shape: [center_num, 3x3x3, 1]."));

        int kernel_size = 3;
        int channels = input_features.dim_size(1);
        int output_pooling_size = output_idx.dim_size(2);
        int kernel_num = kernel_size * kernel_size * kernel_size;
        int center_num = output_idx.dim_size(0);

//        printf("******************input shape = %d************************\n", input_point_num);
//        printf("******************output shape = %d************************\n", kernel_num);

        Tensor* output_features = nullptr;
        auto output_features_shape = TensorShape({center_num, kernel_num, channels});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_features_shape, &output_features));
        float* output_features_ptr = output_features->template flat<float>().data();
        cudaMemset(output_features_ptr, 0, center_num*kernel_num*channels*sizeof(float));

        voxel_sampling_feature_gpu_launcher(center_num, kernel_num, channels, padding_value,
                                            output_pooling_size,
                                            input_features_ptr,
                                            output_idx_ptr,
                                            output_features_ptr);

    }
private:
    float padding_value;
}; // OpKernel
REGISTER_KERNEL_BUILDER(Name("VoxelSamplingFeatureOp").Device(DEVICE_GPU), VoxelSamplingFeatureOp);



REGISTER_OP("VoxelSamplingFeatureGradOp")
    .Input("input_features: float32")
    .Input("output_idx: int32")
    .Input("output_features_grad: float32")
    .Output("input_features_grad: float32")
    .SetShapeFn([](InferenceContext* c){
        c->set_output(0, c->input(0));
        return Status::OK();
    }); // InferenceContext


void voxel_sampling_feature_grad_gpu_launcher(int center_num, int kernel_num, int channels,
                                              int output_pooling_size,
                                              const int* output_idx,
                                              const float* output_features_grad,
                                              float* input_features_grad);

class VoxelSamplingFeatureGradOp: public OpKernel {
public:
    explicit VoxelSamplingFeatureGradOp(OpKernelConstruction* context): OpKernel(context) {}
    void Compute(OpKernelContext* context) override {

        const Tensor& input_features = context->input(0);
        auto input_features_ptr = input_features.template flat<float>().data();
        OP_REQUIRES(context, input_features.dims()==2 && input_features.dim_size(1) > 0,
                    errors::InvalidArgument("VoxelSamplingGradOp expects input features in shape: [point_nums, channels(>0)]."));

        const Tensor& output_idx = context->input(1);
        auto output_idx_ptr = output_idx.template flat<int>().data();
        OP_REQUIRES(context, output_idx.dims()==3 && output_idx.dim_size(2) > 0,
                    errors::InvalidArgument("VoxelSamplingGradOp expects output_idx in shape: [ncenters, kernel_size*3, channels(>0)]."));

        const Tensor& output_features_grad = context->input(2);
        auto output_features_grad_ptr = output_features_grad.template flat<float>().data();
        OP_REQUIRES(context, output_features_grad.dims()==3 && output_features_grad.dim_size(2) > 0,
                    errors::InvalidArgument("VoxelSamplingGradOp expects output_features_grad in shape: [point_nums, kernel_size*3, channels(>0)]."));
        OP_REQUIRES(context, output_idx.dim_size(0) == output_features_grad.dim_size(0) &&
                             output_idx.dim_size(1) == output_features_grad.dim_size(1),
                             errors::InvalidArgument("VoxelSamplingGradOp needs output_features and output_idx has the same length."));

        int input_point_num = input_features.dim_size(0);
        int channels = input_features.dim_size(1);
        int output_pooling_size = output_idx.dim_size(2);
        int center_num = output_idx.dim_size(0);
        int kernel_num = output_idx.dim_size(1);

        Tensor* input_features_grad = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{input_point_num, channels}, &input_features_grad));
        auto input_features_grad_ptr = input_features_grad->template flat<float>().data();
        cudaMemset(input_features_grad_ptr, 0.f, input_point_num*channels*sizeof(float));

        voxel_sampling_feature_grad_gpu_launcher(center_num, kernel_num, channels,
                                                 output_pooling_size,
                                                 output_idx_ptr,
                                                 output_features_grad_ptr,
                                                 input_features_grad_ptr);
    }
};
REGISTER_KERNEL_BUILDER(Name("VoxelSamplingFeatureGradOp").Device(DEVICE_GPU), VoxelSamplingFeatureGradOp);
