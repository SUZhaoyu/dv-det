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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("VoxelSamplingBinaryOp")
    .Input("input_coors: float32")
    .Input("input_features: float32")
    .Input("input_voxel_idx: int64")
    .Input("input_num_list: int32")
    .Input("center_coors: float32")
    .Input("center_num_list: int32")
    .Output("output_features: float32") // [center_coors.shape[0], kernel_size ** 3, channels]
    .Output("output_idx: int32") // [center_coors.shape[0], kernel_size ** 3, channels]
    .Attr("dimension: list(float)")
    .Attr("resolution: float")
    .Attr("padding_value: float")
    .SetShapeFn([](InferenceContext* c){
        ShapeHandle input_features_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input_features_shape));
        ShapeHandle center_coors_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &center_coors_shape));

        int kernel_size = 3;

        DimensionHandle output_ncenter = c->Dim(center_coors_shape, 0);
        DimensionHandle channels = c->Dim(input_features_shape, 1);

        // The output shape during the shape inference stage is pseudo.
        ShapeHandle output_feature_shape = c->MakeShape({output_ncenter, kernel_size*kernel_size*kernel_size, channels});
        ShapeHandle output_idx_shape = c->MakeShape({output_ncenter, kernel_size*kernel_size*kernel_size, 1});

        c->set_output(0, output_feature_shape); // output_features
        c->set_output(1, output_idx_shape); // output_idx

        return Status::OK();

    }); // InferenceContext


REGISTER_OP("VoxelSamplingBinaryGradOp")
    .Input("input_features: float32")
    .Input("output_idx: int32")
    .Input("output_features_grad: float32")
    .Output("input_features_grad: float32")
    .SetShapeFn([](InferenceContext* c){
        c->set_output(0, c->input(0));
        return Status::OK();
    }); // InferenceContext


void voxel_sampling_binary_gpu_launcher(int batch_size, int input_npoint, int channels,
                               int output_ncenter, int kernel_size,
                               float dim_w, float dim_l, float dim_h,
                               float resolution, float padding,
                               const float* input_coors,
                               const float* input_features,
                               const long long* input_voxel_idx,
                               const int* input_num_list,
                               const float* center_coors,
                               const int* center_num_list,
                               int* input_accu_list,
                               int* center_accu_list,
                               float* output_features,
                               int* output_idx);

class VoxelSamplingBinaryOp: public OpKernel {
public:
    explicit VoxelSamplingBinaryOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("padding_value", &padding_value));
        OP_REQUIRES_OK(context, context->GetAttr("resolution", &resolution));
        OP_REQUIRES_OK(context, context->GetAttr("dimension", &dimension));
        OP_REQUIRES(context, resolution > 0,
                    errors::InvalidArgument("Resolution has to be greater than 0"));
        OP_REQUIRES(context, dimension.size() == 3,
                    errors::InvalidArgument("Dimension has to be 3-D for Voxel Sample Operation."));
    }
    void Compute(OpKernelContext* context) override {

        const Tensor& input_coors = context->input(0);
        auto input_coors_ptr = input_coors.template flat<float>().data();
        OP_REQUIRES(context, input_coors.dims()==2 && input_coors.dim_size(1)==3,
                    errors::InvalidArgument("Voxel Sample Op expects input_coors in shape: [npoints, 3]."));

        const Tensor& input_features = context->input(1);
        auto input_features_ptr = input_features.template flat<float>().data();
        OP_REQUIRES(context, input_features.dims()==2 && input_features.dim_size(1) > 0,
                    errors::InvalidArgument("Voxel Sample Op expects input_features in shape: [npoints, channels(>0)]."));

        const Tensor& input_voxel_idx = context->input(2);
        auto input_voxel_idx_ptr = input_voxel_idx.template flat<long long>().data();
        OP_REQUIRES(context, input_voxel_idx.dims()==1,
                    errors::InvalidArgument("Voxel Sample Op expects input_voxel_idx in shape: [npoints]."));

        const Tensor& input_num_list = context->input(3);
        auto input_num_list_ptr = input_num_list.template flat<int>().data();
        OP_REQUIRES(context, input_num_list.dims()==1,
                    errors::InvalidArgument("Voxel Sample Op expects input_num_list in shape: [batch_size]."));

        const Tensor& center_coors = context->input(4);
        auto center_coors_ptr = center_coors.template flat<float>().data();
        OP_REQUIRES(context, center_coors.dims()==2 && center_coors.dim_size(1)==3,
                    errors::InvalidArgument("Voxel Sample Op expects center coors in shape: [ncenters, 3]."));

        const Tensor& center_num_list = context->input(5);
        auto center_num_list_ptr = center_num_list.template flat<int>().data();
        OP_REQUIRES(context, center_num_list.dims()==1,
                    errors::InvalidArgument("Voxel Sample Op expects center_num_list in shape: [batch_size]."));

        int kernel_size = 3;
        int input_npoint = input_coors.dim_size(0);
        int output_ncenter = center_coors.dim_size(0);
        int batch_size = input_num_list.dim_size(0);
        int channels = input_features.dim_size(1);
        int ngrid = kernel_size * kernel_size * kernel_size;

        int batch_byte_size = batch_size * sizeof(int);
        int* input_num_list_ptr_host = (int*)malloc(batch_byte_size);
        int* center_num_list_ptr_host = (int*)malloc(batch_byte_size);
        int* input_accu_list_ptr_host = (int*)malloc(batch_byte_size);
        int* center_accu_list_ptr_host = (int*)malloc(batch_byte_size);
        input_accu_list_ptr_host[0] = 0;
        center_accu_list_ptr_host[0] = 0;
        cudaMemcpy(input_num_list_ptr_host, input_num_list_ptr, batch_byte_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(center_num_list_ptr_host, center_num_list_ptr, batch_byte_size, cudaMemcpyDeviceToHost);

        for (int b=1; b<batch_size; b++) {
            input_accu_list_ptr_host[b] = input_accu_list_ptr_host[b-1] + input_num_list_ptr_host[b-1];
            center_accu_list_ptr_host[b] = center_accu_list_ptr_host[b-1] + center_num_list_ptr_host[b-1];
//            printf("%d=%d+%d, %d\n", center_accu_list_ptr_host[b], center_accu_list_ptr_host[b-1], center_num_list_ptr_host[b-1], center_num_list_ptr_host[b]);
        }

        int count = 0;
        for (int b=0; b<batch_size; b++) {
            count += center_num_list_ptr_host[b];
        }
        if (count != output_ncenter)
            printf("*****************Mismatch Dimension: %d vs. %d; input_npoint=%d\n", output_ncenter, count, input_npoint);

        Tensor input_accu_list;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{batch_size},
                                                       &input_accu_list));
        int* input_accu_list_ptr = input_accu_list.template flat<int>().data();
        cudaMemcpy(input_accu_list_ptr, input_accu_list_ptr_host, batch_byte_size, cudaMemcpyHostToDevice);

        Tensor center_accu_list;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{batch_size},
                                                       &center_accu_list));
        int* center_accu_list_ptr = center_accu_list.template flat<int>().data();
        cudaMemcpy(center_accu_list_ptr, center_accu_list_ptr_host, batch_byte_size, cudaMemcpyHostToDevice);

        Tensor* output_features = nullptr;
        auto output_features_shape = TensorShape({output_ncenter, ngrid, channels});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_features_shape, &output_features));
        float* output_features_ptr = output_features->template flat<float>().data();
//        cudaMemset(output_features_ptr, padding_value, output_ncenter*ngrid*channels*sizeof(float));

        Tensor* output_idx = nullptr;
        auto output_idx_shape = TensorShape({output_ncenter, ngrid, 1});
        OP_REQUIRES_OK(context, context->allocate_output(1, output_idx_shape, &output_idx));
        int* output_idx_ptr = output_idx->template flat<int>().data();
//        printf("VoxelSamplingBinaryOutputIdx Shape=[%d, %d, %d]\n", output_ncenter, ngrid, 1);
//        cudaMemset(output_idx_ptr, -1, output_ncenter*ngrid*channels*sizeof(int));

        voxel_sampling_binary_gpu_launcher(batch_size, input_npoint, channels,
                                  output_ncenter, kernel_size,
                                  dimension[0], dimension[1], dimension[2],
                                  resolution, padding_value,
                                  input_coors_ptr,
                                  input_features_ptr,
                                  input_voxel_idx_ptr,
                                  input_num_list_ptr,
                                  center_coors_ptr,
                                  center_num_list_ptr,
                                  input_accu_list_ptr,
                                  center_accu_list_ptr,
                                  output_features_ptr,
                                  output_idx_ptr);
        free(input_num_list_ptr_host);
        free(center_num_list_ptr_host);
        free(input_accu_list_ptr_host);
        free(center_accu_list_ptr_host);
    }
private:
    float padding_value, resolution;
    std::vector<float> dimension;
}; // OpKernel
REGISTER_KERNEL_BUILDER(Name("VoxelSamplingBinaryOp").Device(DEVICE_GPU), VoxelSamplingBinaryOp);




void voxel_sampling_binary_grad_gpu_launcher(int output_ncenter, int ngrid, int channels,
                                    const int* output_idx,
                                    const float* output_features_grad,
                                    float* input_features_grad);

class VoxelSamplingBinaryGradOp: public OpKernel {
public:
    explicit VoxelSamplingBinaryGradOp(OpKernelConstruction* context): OpKernel(context) {}
    void Compute(OpKernelContext* context) override {

        const Tensor& input_features = context->input(0);
        auto input_features_ptr = input_features.template flat<float>().data();
        OP_REQUIRES(context, input_features.dims()==2 && input_features.dim_size(1) > 0,
                    errors::InvalidArgument("Voxel Sample Op expects input features in shape: [npoints, channels(>0)]."));

        const Tensor& output_idx = context->input(1);
        auto output_idx_ptr = output_idx.template flat<int>().data();
        OP_REQUIRES(context, output_idx.dims()==3 && output_idx.dim_size(2)==1,
                    errors::InvalidArgument("Voxel Sample Op expects output_idx in shape: [ncenters, kernel_size*3, 1]."));
//        printf("VoxelSamplingBinaryGradOutputIdx Shape=[%d, %d, %d]\n", output_idx.dim_size(0), output_idx.dim_size(1), output_idx.dim_size(2));

        const Tensor& output_features_grad = context->input(2);
        auto output_features_grad_ptr = output_features_grad.template flat<float>().data();
        OP_REQUIRES(context, output_features_grad.dims()==3 && output_features_grad.dim_size(2) > 0,
                    errors::InvalidArgument("Voxel Sample Op expects output_features_grad in shape: [npoints, kernel_size*3, channels(>0)]."));
        OP_REQUIRES(context, output_idx.dim_size(0)==output_features_grad.dim_size(0) &&
                             output_idx.dim_size(1)==output_features_grad.dim_size(1) &&
                             output_features_grad.dim_size(2)==input_features.dim_size(1),
                             errors::InvalidArgument("Output_idx and output_features_grad has to be of the same shape."));

        int input_npoint = input_features.dim_size(0);
        int channels = input_features.dim_size(1);
        int output_ncenter = output_idx.dim_size(0);
        int ngrid = output_idx.dim_size(1);

//        int* output_idx_host_ptr = (int*)malloc(output_ncenter*ngrid*sizeof(int));
//        cudaMemcpy(output_idx_host_ptr, output_idx_ptr, output_ncenter*ngrid*sizeof(int), cudaMemcpyDeviceToHost);
//        for (int i=0; i<output_ncenter*ngrid; i++) {
//            if (output_idx_host_ptr[i] > 100000)
//                printf("output_idx=%d\n", output_idx_host_ptr[i]);
//        }
//        free(output_idx_host_ptr);

        Tensor* input_features_grad = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{input_npoint, channels}, &input_features_grad));
        auto input_features_grad_ptr = input_features_grad->template flat<float>().data();
        cudaMemset(input_features_grad_ptr, 0.f, input_npoint*channels*sizeof(float));

        voxel_sampling_binary_grad_gpu_launcher(output_ncenter, ngrid, channels,
                                       output_idx_ptr,
                                       output_features_grad_ptr,
                                       input_features_grad_ptr);
    }
};
REGISTER_KERNEL_BUILDER(Name("VoxelSamplingBinaryGradOp").Device(DEVICE_GPU), VoxelSamplingBinaryGradOp);
