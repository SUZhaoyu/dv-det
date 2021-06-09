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

REGISTER_OP("DenseVoxelizationOp")
    .Input("input_coors: float32")
    .Input("input_features: float32")
    .Input("input_num_list: int32")
    .Output("output_features: float32") // [center_coors.shape[0], kernel_size ** 3, channels]
    .Output("output_idx: int32")
    .Attr("dimension: list(float)")
    .Attr("resolution: list(float)")
    .SetShapeFn([](InferenceContext* c){
        ShapeHandle input_features_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input_features_shape));

        ShapeHandle input_num_list_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &input_num_list_shape));

        DimensionHandle batch_size = c->Dim(input_num_list_shape, 0);
        DimensionHandle channels = c->Dim(input_features_shape, 1);

        std::vector<float> resolution;
        std::vector<float> dimension;
        TF_RETURN_IF_ERROR(c->GetAttr("resolution", &resolution));
        TF_RETURN_IF_ERROR(c->GetAttr("dimension", &dimension));
        int output_w = (int)ceil(dimension[0] / resolution[0]);
        int output_l = (int)ceil(dimension[1] / resolution[1]);
        int output_h = (int)ceil(dimension[2] / resolution[2]);

        // The output shape during the shape inference stage is pseudo.
        ShapeHandle output_features_shape = c->MakeShape({batch_size, output_w, output_l, output_h, channels});
        ShapeHandle output_idx_shape = c->MakeShape({batch_size, output_w, output_l, output_h});

        c->set_output(0, output_features_shape); // output_idx
        c->set_output(1, output_idx_shape); // output_idx

        return Status::OK();

    }); // InferenceContext


void dense_voxelization_gpu_launcher(int batch_size, int input_point_num, int channels,
                                     std::vector<float> resolution, std::vector<int> output_size,
                                     const float* input_coors,
                                     const float* input_features,
                                     const int* input_num_list,
                                     int* input_accu_list,
                                     int* count_buffer,
                                     float* output_features,
                                     int* output_idx);

class DenseVoxelizationOp: public OpKernel {
public:
    explicit DenseVoxelizationOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("resolution", &resolution));
        OP_REQUIRES_OK(context, context->GetAttr("dimension", &dimension));
        OP_REQUIRES(context, resolution.size() == 3,
                    errors::InvalidArgument("Resolution has to be 3-D for DenseVoxelizationOp"));
        OP_REQUIRES(context, dimension.size() == 3,
                    errors::InvalidArgument("Dimension has to be 3-D for DenseVoxelizationOp."));
    }
    void Compute(OpKernelContext* context) override {

        const Tensor& input_coors = context->input(0);
        auto input_coors_ptr = input_coors.template flat<float>().data();
        OP_REQUIRES(context, input_coors.dims()==2 && input_coors.dim_size(1)==3,
                    errors::InvalidArgument("DenseVoxelizationOp expects input_coors in shape: [point_nums, 3]."));

        const Tensor& input_features = context->input(1);
        auto input_features_ptr = input_features.template flat<float>().data();
        OP_REQUIRES(context, input_features.dims()==2 && input_features.dim_size(1)>0,
                    errors::InvalidArgument("DenseVoxelizationOp expects input_coors in shape: [point_nums, channels]."));

        const Tensor& input_num_list = context->input(2);
        auto input_num_list_ptr = input_num_list.template flat<int>().data();
        OP_REQUIRES(context, input_num_list.dims()==1,
                    errors::InvalidArgument("DenseVoxelizationOp expects input_num_list in shape: [batch_size]."));

        int batch_size = input_num_list.dim_size(0);
        int input_point_num = input_features.dim_size(0);
        int channels = input_features.dim_size(1);
        int output_w = (int)ceil(dimension[0] / resolution[0]);
        int output_l = (int)ceil(dimension[1] / resolution[1]);
        int output_h = (int)ceil(dimension[2] / resolution[2]);
        std::vector<int> output_size {output_w, output_l, output_h};


        int batch_byte_size = batch_size * sizeof(int);
        int* input_num_list_ptr_host = (int*)malloc(batch_byte_size);
        int* input_accu_list_ptr_host = (int*)malloc(batch_byte_size);
        input_accu_list_ptr_host[0] = 0;
        cudaMemcpy(input_num_list_ptr_host, input_num_list_ptr, batch_byte_size, cudaMemcpyDeviceToHost);

        for (int b=1; b<batch_size; b++) {
            input_accu_list_ptr_host[b] = input_accu_list_ptr_host[b-1] + input_num_list_ptr_host[b-1];
        }
//
        Tensor input_accu_list;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{batch_size},
                                                       &input_accu_list));
        int* input_accu_list_ptr = input_accu_list.template flat<int>().data();
        cudaMemcpy(input_accu_list_ptr, input_accu_list_ptr_host, batch_byte_size, cudaMemcpyHostToDevice);

        Tensor count_buffer;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{batch_size, output_w, output_l, output_h},
                                                       &count_buffer));
        int* count_buffer_ptr = count_buffer.template flat<int>().data();
        cudaMemset(count_buffer_ptr, 0, batch_size*output_w*output_l*output_h*sizeof(int));

        Tensor* output_features = nullptr;
        auto output_features_shape = TensorShape({batch_size, output_w, output_l, output_h, channels});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_features_shape, &output_features));
        float* output_features_ptr = output_features->template flat<float>().data();
        cudaMemset(output_features_ptr, 0, batch_size*output_w*output_l*output_h*channels*sizeof(float));

        Tensor* output_idx = nullptr;
        auto output_idx_shape = TensorShape({batch_size, output_w, output_l, output_h});
        OP_REQUIRES_OK(context, context->allocate_output(1, output_idx_shape, &output_idx));
        int* output_idx_ptr = output_idx->template flat<int>().data();
        cudaMemset(output_idx_ptr, 0xEF, batch_size*output_w*output_l*output_h*sizeof(int));

        dense_voxelization_gpu_launcher(batch_size, input_point_num, channels,
                                        resolution, output_size,
                                        input_coors_ptr,
                                        input_features_ptr,
                                        input_num_list_ptr,
                                        input_accu_list_ptr,
                                        count_buffer_ptr,
                                        output_features_ptr,
                                        output_idx_ptr);

        free(input_num_list_ptr_host);
    }
private:
    std::vector<float> resolution;
    std::vector<float> dimension;
}; // OpKernel
REGISTER_KERNEL_BUILDER(Name("DenseVoxelizationOp").Device(DEVICE_GPU), DenseVoxelizationOp);


REGISTER_OP("DenseVoxelizationGradOp")
    .Input("input_features: float32")
    .Input("output_idx: int32")
    .Input("output_features_grad: float32")
    .Output("input_features_grad: float32")
    .SetShapeFn([](InferenceContext* c){
        c->set_output(0, c->input(0));
        return Status::OK();
    }); // InferenceContext


void dense_voxelization_grad_gpu_launcher(int batch_size, int channels, std::vector<int> output_size,
                                          const float* output_features_grad,
                                          const int* output_idx,
                                          float* input_features_grad);

class DenseVoxelizationGradOp: public OpKernel {
public:
    explicit DenseVoxelizationGradOp(OpKernelConstruction* context): OpKernel(context) {}
    void Compute(OpKernelContext* context) override {

        const Tensor& input_features = context->input(0);
        auto input_features_ptr = input_features.template flat<float>().data();
        OP_REQUIRES(context, input_features.dims()==2 && input_features.dim_size(1) > 0,
                    errors::InvalidArgument("DenseVoxelizationGradOp expects input features in shape: [point_nums, channels(>0)]."));

        const Tensor& output_idx = context->input(1);
        auto output_idx_ptr = output_idx.template flat<int>().data();
        OP_REQUIRES(context, output_idx.dims()==5,
                    errors::InvalidArgument("DenseVoxelizationGradOp expects output_idx in shape: [batch_size, output_w, output_l, output_h]."));

        const Tensor& output_features_grad = context->input(2);
        auto output_features_grad_ptr = output_features_grad.template flat<float>().data();
        OP_REQUIRES(context, output_features_grad.dims()==5 && output_features_grad.dim_size(4) == input_features.dim_size(1),
                    errors::InvalidArgument("DenseVoxelizationGradOp expects output_features_grad in shape: [batch_size, output_w, output_l, output_h, channels(>0)]."));


        int input_point_num = input_features.dim_size(0);
        int channels = input_features.dim_size(1);
        int batch_size = output_idx.dim_size(0);
        int output_w = output_idx.dim_size(1);
        int output_l = output_idx.dim_size(2);
        int output_h = output_idx.dim_size(3);
        std::vector<int> output_size {output_w, output_l, output_h};

        Tensor* input_features_grad = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{input_point_num, channels}, &input_features_grad));
        auto input_features_grad_ptr = input_features_grad->template flat<float>().data();
        cudaMemset(input_features_grad_ptr, 0.f, input_point_num*channels*sizeof(float));

        dense_voxelization_grad_gpu_launcher(batch_size, channels, output_size,
                                             output_features_grad_ptr,
                                             output_idx_ptr,
                                             input_features_grad_ptr);
    }
};
REGISTER_KERNEL_BUILDER(Name("DenseVoxelizationGradOp").Device(DEVICE_GPU), DenseVoxelizationGradOp);
