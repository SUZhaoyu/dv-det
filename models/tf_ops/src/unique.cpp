/* Unique Operation
 * GPU implementation of unique operation.
 * Created by Zhaoyu SU
 * All Rights Reserved. Nov., 2020.
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

REGISTER_OP("UniqueOp")
    .Input("input_voxel_ids: int64")
    .Input("input_point_ids: int32")
    .Output("output_point_ids: int32") // [center_coors.shape[0], kernel_size ** 3, channels]
    .SetShapeFn([](InferenceContext* c){
        ShapeHandle input_point_ids_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input_point_ids_shape));
        c->set_output(0, input_point_ids_shape); // output_features

        return Status::OK();

    }); // InferenceContext

int unique_gpu_launcher(long long* input_voxel_ids_temp,
                               int* input_point_ids_temp,
                               int input_npoint);

class UniqueOp: public OpKernel {
public:
    explicit UniqueOp(OpKernelConstruction* context): OpKernel(context) {}
    void Compute(OpKernelContext* context) override {

        const Tensor& input_voxel_ids = context->input(0);
        auto input_voxel_ids_ptr = input_voxel_ids.template flat<long long>().data();
        OP_REQUIRES(context, input_voxel_ids.dims()==1,
                    errors::InvalidArgument("Unique Op expects input_voxel_ids in 1-D"));

        const Tensor& input_point_ids = context->input(1);
        auto input_point_ids_ptr = input_point_ids.template flat<int>().data();
        OP_REQUIRES(context, input_point_ids.dims()==1 && input_voxel_ids.dim_size(0)==input_point_ids.dim_size(0),
                    errors::InvalidArgument("Unique Op expects input_point_ids in 1-D and has the same length as input_voxel_ids."));

        int input_npoint = input_voxel_ids.dim_size(0);

        Tensor input_voxel_ids_temp;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<long long>::value,
                                                       TensorShape{input_npoint},
                                                       &input_voxel_ids_temp));
        long long* input_voxel_ids_temp_ptr = input_voxel_ids_temp.template flat<long long>().data();
        cudaMemcpy(input_voxel_ids_temp_ptr, input_voxel_ids_ptr, input_npoint * sizeof(long long), cudaMemcpyDeviceToDevice);

        Tensor input_point_ids_temp;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{input_npoint},
                                                       &input_point_ids_temp));
        int* input_point_ids_temp_ptr = input_point_ids_temp.template flat<int>().data();
        cudaMemcpy(input_point_ids_temp_ptr, input_point_ids_ptr, input_npoint * sizeof(int), cudaMemcpyDeviceToDevice);

        int output_npoint = unique_gpu_launcher(input_voxel_ids_temp_ptr,
                                                input_point_ids_temp_ptr,
                                                input_npoint);

        Tensor* output_point_ids = nullptr;
        auto output_point_ids_shape = TensorShape({output_npoint});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_point_ids_shape, &output_point_ids));
        int* output_point_ids_ptr = output_point_ids->template flat<int>().data();
        cudaMemcpy(output_point_ids_ptr, input_point_ids_temp_ptr, output_npoint * sizeof(int), cudaMemcpyDeviceToDevice);




//        Tensor output_ids_temp;
//        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<long long>::value,
//                                                       TensorShape{input_npoint},
//                                                       &output_ids_temp));
//        long long* output_ids_temp_ptr = output_ids_temp.template flat<long long>().data();
//        cudaMemcpy(output_ids_temp_ptr, input_ids_ptr, input_npoint * sizeof(long long), cudaMemcpyDeviceToDevice);
    }
};
REGISTER_KERNEL_BUILDER(Name("UniqueOp").Device(DEVICE_GPU), UniqueOp);
