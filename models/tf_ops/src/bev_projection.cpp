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

REGISTER_OP("BevProjectionOp")
    .Input("input_coors: float32")
    .Input("input_features: float32")
    .Input("input_num_list: int32")
    .Output("output_features: float32") // [center_coors.shape[0], kernel_size ** 3, channels]
    .Output("output_idx: int32")
    .Attr("dimension: list(float)")
    .Attr("resolution: float")
    .Attr("buffer_size: int")
    .SetShapeFn([](InferenceContext* c){
        ShapeHandle input_features_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input_features_shape));

        ShapeHandle input_num_list_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &input_num_list_shape));

        DimensionHandle batch_size = c->Dim(input_num_list_shape, 0);
        DimensionHandle channels = c->Dim(input_features_shape, 1);

        int buffer_size;
        float resolution;
        std::vector<float> dimension;
        TF_RETURN_IF_ERROR(c->GetAttr("buffer_size", &buffer_size));
        TF_RETURN_IF_ERROR(c->GetAttr("resolution", &resolution));
        TF_RETURN_IF_ERROR(c->GetAttr("dimension", &dimension));
        int output_w = (int)ceil(dimension[0] / resolution);
        int output_l = (int)ceil(dimension[1] / resolution);

        // The output shape during the shape inference stage is pseudo.
        ShapeHandle output_features_shape = c->MakeShape({batch_size, output_w, output_l, channels});
        ShapeHandle output_idx_shape = c->MakeShape({batch_size, output_w, output_l, channels});

        c->set_output(0, output_features_shape); // output_idx
        c->set_output(1, output_idx_shape); // output_idx

        return Status::OK();

    }); // InferenceContext


void bev_projection_gpu_launcher(int batch_size, int input_point_num, int channels,
                                 int output_w, int output_l, float resolution, int buffer_size,
                                 const float* input_coors,
                                 const float* input_features,
                                 const int* input_num_list,
                                 int* input_accu_list,
                                 int* count_buffer,
                                 int* idx_buffer,
                                 float* output_features,
                                 int* output_idx);

class BevProjectionOp: public OpKernel {
public:
    explicit BevProjectionOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("resolution", &resolution));
        OP_REQUIRES_OK(context, context->GetAttr("dimension", &dimension));
        OP_REQUIRES_OK(context, context->GetAttr("buffer_size", &buffer_size));
        OP_REQUIRES(context, resolution > 0,
                    errors::InvalidArgument("Resolution has to be greater than 0"));
        OP_REQUIRES(context, dimension.size() == 3,
                    errors::InvalidArgument("Dimension has to be 3-D for Voxel Sample Operation."));
    }
    void Compute(OpKernelContext* context) override {

        const Tensor& input_coors = context->input(0);
        auto input_coors_ptr = input_coors.template flat<float>().data();
        OP_REQUIRES(context, input_coors.dims()==2 && input_coors.dim_size(1)==3,
                    errors::InvalidArgument("BevProjectionOp expects input_coors in shape: [point_nums, 3]."));

        const Tensor& input_features = context->input(1);
        auto input_features_ptr = input_features.template flat<float>().data();
        OP_REQUIRES(context, input_features.dims()==2 && input_features.dim_size(1)>0,
                    errors::InvalidArgument("BevProjectionOp expects input_coors in shape: [point_nums, channels]."));

        const Tensor& input_num_list = context->input(2);
        auto input_num_list_ptr = input_num_list.template flat<int>().data();
        OP_REQUIRES(context, input_num_list.dims()==1,
                    errors::InvalidArgument("BevProjectionOp expects input_num_list in shape: [batch_size]."));

        int batch_size = input_num_list.dim_size(0);
        int input_point_num = input_features.dim_size(0);
        int channels = input_features.dim_size(1);
        int output_w = (int)ceil(dimension[0] / resolution);
        int output_l = (int)ceil(dimension[1] / resolution);


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
                                                       TensorShape{batch_size, output_w, output_l},
                                                       &count_buffer));
        int* count_buffer_ptr = count_buffer.template flat<int>().data();
        cudaMemset(count_buffer_ptr, 0, batch_size*output_w*output_l*sizeof(int));

        Tensor idx_buffer;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{batch_size, output_w, output_l, buffer_size},
                                                       &idx_buffer));
        int* idx_buffer_ptr = idx_buffer.template flat<int>().data();
        cudaMemset(idx_buffer_ptr, 0xEF, batch_size*output_w*output_l*buffer_size*sizeof(int));

        Tensor* output_features = nullptr;
        auto output_features_shape = TensorShape({batch_size, output_w, output_l, channels});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_features_shape, &output_features));
        float* output_features_ptr = output_features->template flat<float>().data();
        cudaMemset(output_features_ptr, 0, batch_size*output_w*output_l*channels*sizeof(float));

        Tensor* output_idx = nullptr;
        auto output_idx_shape = TensorShape({batch_size, output_w, output_l, channels});
        OP_REQUIRES_OK(context, context->allocate_output(1, output_idx_shape, &output_idx));
        int* output_idx_ptr = output_idx->template flat<int>().data();
        cudaMemset(output_idx_ptr, 0xEF, batch_size*output_w*output_l*channels*sizeof(int));

        bev_projection_gpu_launcher(batch_size, input_point_num, channels,
                                    output_w, output_l, resolution, buffer_size,
                                    input_coors_ptr,
                                    input_features_ptr,
                                    input_num_list_ptr,
                                    input_accu_list_ptr,
                                    count_buffer_ptr,
                                    idx_buffer_ptr,
                                    output_features_ptr,
                                    output_idx_ptr);

        free(input_num_list_ptr_host);
    }
private:
    float resolution;
    int buffer_size;
    std::vector<float> dimension;
}; // OpKernel
REGISTER_KERNEL_BUILDER(Name("BevProjectionOp").Device(DEVICE_GPU), BevProjectionOp);


REGISTER_OP("BevProjectionGradOp")
    .Input("input_features: float32")
    .Input("output_idx: int32")
    .Input("output_features_grad: float32")
    .Output("input_features_grad: float32")
    .SetShapeFn([](InferenceContext* c){
        c->set_output(0, c->input(0));
        return Status::OK();
    }); // InferenceContext


void bev_projection_grad_gpu_launcher(int batch_size, int output_w, int output_l, int channels,
                                      const float* output_features_grad,
                                      const int* output_idx,
                                      float* input_features_grad);

class BevProjectionGradOp: public OpKernel {
public:
    explicit BevProjectionGradOp(OpKernelConstruction* context): OpKernel(context) {}
    void Compute(OpKernelContext* context) override {

        const Tensor& input_features = context->input(0);
        auto input_features_ptr = input_features.template flat<float>().data();
        OP_REQUIRES(context, input_features.dims()==2 && input_features.dim_size(1) > 0,
                    errors::InvalidArgument("BevProjectionGradOp expects input features in shape: [point_nums, channels(>0)]."));

        const Tensor& output_idx = context->input(1);
        auto output_idx_ptr = output_idx.template flat<int>().data();
        OP_REQUIRES(context, output_idx.dims()==4 && output_idx.dim_size(3) == input_features.dim_size(1),
                    errors::InvalidArgument("BevProjectionGradOp expects output_idx in shape: [batch_size, output_w, output_l, channels(>0)]."));

        const Tensor& output_features_grad = context->input(2);
        auto output_features_grad_ptr = output_features_grad.template flat<float>().data();
        OP_REQUIRES(context, output_features_grad.dims()==4 && output_features_grad.dim_size(3) == input_features.dim_size(1),
                    errors::InvalidArgument("BevProjectionGradOp expects output_features_grad in shape: [batch_size, output_w, output_l, channels(>0)]."));


        int input_point_num = input_features.dim_size(0);
        int channels = input_features.dim_size(1);
        int batch_size = output_idx.dim_size(0);
        int output_w = output_idx.dim_size(1);
        int output_l = output_idx.dim_size(2);

        Tensor* input_features_grad = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{input_point_num, channels}, &input_features_grad));
        auto input_features_grad_ptr = input_features_grad->template flat<float>().data();
        cudaMemset(input_features_grad_ptr, 0.f, input_point_num*channels*sizeof(float));

        bev_projection_grad_gpu_launcher(batch_size, output_w, output_l, channels,
                                         output_features_grad_ptr,
                                         output_idx_ptr,
                                         input_features_grad_ptr);
    }
};
REGISTER_KERNEL_BUILDER(Name("BevProjectionGradOp").Device(DEVICE_GPU), BevProjectionGradOp);
