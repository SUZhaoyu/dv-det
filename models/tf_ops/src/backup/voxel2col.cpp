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
#include <list>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("VoxelToColOp")
    .Input("input_voxels: float32")
    .Output("output_voxels: float32") // [center_coors.shape[0], voxel_size ** 3, channels]
    .Output("output_idx: int32") // [center_coors.shape[0], voxel_size ** 3, channels]
    .Attr("kernel_size: int")
    .Attr("channels: int")
    .SetShapeFn([](InferenceContext* c){
        ShapeHandle input_voxels_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input_voxels_shape));

        int kernel_size, channels;
        TF_RETURN_IF_ERROR(c->GetAttr("kernel_size", &kernel_size));
        TF_RETURN_IF_ERROR(c->GetAttr("channels", &channels));

        DimensionHandle input_num = c->Dim(input_voxels_shape, 0);
        auto input_voxel_num = c->Value(c->Dim(input_voxels_shape, 1));
        auto output_voxel_num = (int)std::pow(cbrt(input_voxel_num) - 2, 3);

        // The output shape during the shape inference stage is pseudo.
        ShapeHandle output_voxels_shape = c->MakeShape({input_num, output_voxel_num, kernel_size * kernel_size * kernel_size * channels});
        ShapeHandle output_idx_shape = c->MakeShape({input_num, output_voxel_num, kernel_size * kernel_size * kernel_size});
//        printf("********************output_voxel_num=%d\n************************", output_voxel_num);
//        InferenceContext::kUnknownDim
        c->set_output(0, output_voxels_shape); // output_voxels_shape
        c->set_output(1, output_idx_shape); // output_idx

        return Status::OK();
    }); // InferenceContext


REGISTER_OP("VoxelToColGradOp")
    .Input("input_voxels: float32")
    .Input("output_idx: int32")
    .Input("output_voxels_grad: float32")
    .Output("input_voxels_grad: float32")
    .SetShapeFn([](InferenceContext* c){
        c->set_output(0, c->input(0));
        return Status::OK();
    }); // InferenceContext


void voxel2col_gpu_launcher(int input_num, int channels, int input_voxel_size, int output_voxel_size, int kernel_size,
                             const float* input_voxels,
                             float* output_voxels,
                             int* output_idx);

class VoxelToColOp: public OpKernel {
public:
    explicit VoxelToColOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("kernel_size", &kernel_size));
        OP_REQUIRES(context, kernel_size % 2 == 1 && kernel_size > 1,
                    errors::InvalidArgument("DenseConvOp expects kernel_size to be an odd number and it has to be greater than 1."));
    }
    void Compute(OpKernelContext* context) override {

        const Tensor& input_voxels = context->input(0);
        auto input_voxels_ptr = input_voxels.template flat<float>().data();
        OP_REQUIRES(context, input_voxels.dims()==3,
                    errors::InvalidArgument("DenseConvOp expects input_voxels in shape: [input_num, voxel_num, input_channels]."));

        int input_num = input_voxels.dim_size(0);
        int input_voxel_num = input_voxels.dim_size(1);
        int channels = input_voxels.dim_size(2);
        int kernel_num = kernel_size*kernel_size*kernel_size;
        int input_voxel_size = (int)round(cbrt(input_voxel_num));
        OP_REQUIRES(context, input_voxel_size * input_voxel_size * input_voxel_size == input_voxel_num,
                    errors::InvalidArgument("Input 3-D dimension of DenseConvOp must be the same, identical dimensions are not currently supported."));
        OP_REQUIRES(context, input_voxel_size >= kernel_size,
                    errors::InvalidArgument("Input_voxel_size: %d of DenseConvOp has to be greater than kernel_size: %d", input_voxel_size, kernel_size));
        int output_voxel_size = input_voxel_size - 2;
        int output_voxel_num = output_voxel_size * output_voxel_size * output_voxel_size;


        Tensor* output_voxels = nullptr;
        auto output_voxels_shape = TensorShape({input_num, output_voxel_num, kernel_num * channels});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_voxels_shape, &output_voxels));
        float* output_voxels_ptr = output_voxels->template flat<float>().data();
//        printf("Voxel2Col Output Voxel Size: %d MB\n", input_num*output_voxel_num*kernel_num*channels*sizeof(float)/1024/1024);
        cudaMemset(output_voxels_ptr, 0., input_num*output_voxel_num*kernel_num*channels*sizeof(float));

        Tensor* output_idx = nullptr;
        auto output_idx_shape = TensorShape({input_num, output_voxel_num, kernel_num});
        OP_REQUIRES_OK(context, context->allocate_output(1, output_idx_shape, &output_idx));
        int* output_idx_ptr = output_idx->template flat<int>().data();
        cudaMemset(output_idx_ptr, 0, input_num*output_voxel_num*kernel_num*sizeof(int));

        voxel2col_gpu_launcher(input_num, channels, input_voxel_size, output_voxel_size, kernel_size,
                                input_voxels_ptr,
                                output_voxels_ptr,
                                output_idx_ptr);

    }
private:
    int kernel_size;
}; // OpKernel
REGISTER_KERNEL_BUILDER(Name("VoxelToColOp").Device(DEVICE_GPU), VoxelToColOp);


void voxel2col_grad_gpu_launcher(int input_num, int output_voxel_num, int kernel_num, int channels,
                                 const float* input_voxels,
                                 const int* output_idx,
                                 const float* output_voxels_grad,
                                 float* input_voxels_grad);

class VoxelToColGradOp: public OpKernel {
public:
    explicit VoxelToColGradOp(OpKernelConstruction* context): OpKernel(context) {}
    void Compute(OpKernelContext* context) override {

        const Tensor& input_voxels = context->input(0);
        auto input_voxels_ptr = input_voxels.template flat<float>().data();
        OP_REQUIRES(context, input_voxels.dims()==3,
                    errors::InvalidArgument("DenseConvGradOp expects input_voxels in shape: [input_num, input_voxel_num, input_channels]."));

        const Tensor& output_idx = context->input(1);
        auto output_idx_ptr = output_idx.template flat<int>().data();
        OP_REQUIRES(context, output_idx.dims()==3,
                    errors::InvalidArgument("DenseConvGradOp expects output_idx in shape: [input_num, output_voxel_num, kernel_num]."));

        const Tensor& output_voxels_grad = context->input(2);
        auto output_voxels_grad_ptr = output_voxels_grad.template flat<float>().data();
        OP_REQUIRES(context, output_voxels_grad.dims()==3,
                    errors::InvalidArgument("DenseConvGradOp expects output_voxels_grad in shape: [input_num, output_voxel_num, kernel_num * channels]."));

        int input_num = input_voxels.dim_size(0);
        int input_voxel_num = input_voxels.dim_size(1);
        int channels = input_voxels.dim_size(2);
        int output_voxel_num = output_idx.dim_size(1);
        int kernel_num = output_idx.dim_size(2);
        OP_REQUIRES(context, input_voxels.dim_size(0) == output_idx.dim_size(0),
                    errors::InvalidArgument("DenseConvGradOp expects input_voxels and output_idx has the same length (input_point_num)."));
        OP_REQUIRES(context, output_voxels_grad.dim_size(2) / kernel_num == channels,
                    errors::InvalidArgument("DenseConvGradOp expects output_voxels_grad in shape: [input_num, output_voxel_num, kernel_num * channels]."));

        Tensor* input_voxels_grad = nullptr;
        auto input_voxels_grad_shape = TensorShape({input_num, input_voxel_num, channels});
        OP_REQUIRES_OK(context, context->allocate_output(0, input_voxels_grad_shape, &input_voxels_grad));
        float* input_voxels_grad_ptr = input_voxels_grad->template flat<float>().data();
        cudaMemset(input_voxels_grad_ptr, 0., input_num*input_voxel_num*channels*sizeof(float));

        voxel2col_grad_gpu_launcher(input_num, output_voxel_num, kernel_num, channels,
                                    input_voxels_ptr,
                                    output_idx_ptr,
                                    output_voxels_grad_ptr,
                                    input_voxels_grad_ptr);

    }
}; // OpKernel
REGISTER_KERNEL_BUILDER(Name("VoxelToColGradOp").Device(DEVICE_GPU), VoxelToColGradOp);
