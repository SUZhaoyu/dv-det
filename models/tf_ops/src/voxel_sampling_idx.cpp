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

REGISTER_OP("VoxelSamplingIdxOp")
    .Input("input_coors: float32")
    .Input("input_num_list: int32")
    .Input("center_coors: float32")
    .Input("center_num_list: int32")
    .Output("output_idx: int32") // [center_coors.shape[0], kernel_size ** 3, channels]
    .Output("valid_idx: int32")
    .Attr("dimension: list(float)")
    .Attr("resolution: list(float)")
    .Attr("grid_buffer_size: int")
    .Attr("output_pooling_size: int")
    .Attr("with_rpn: bool")
    .SetShapeFn([](InferenceContext* c){
        ShapeHandle center_coors_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &center_coors_shape));

        int kernel_size = 3;

        DimensionHandle center_num = c->Dim(center_coors_shape, 0);
        int output_pooling_size;
        TF_RETURN_IF_ERROR(c->GetAttr("output_pooling_size", &output_pooling_size));

        // The output shape during the shape inference stage is pseudo.
        ShapeHandle output_idx_shape = c->MakeShape({center_num, kernel_size * kernel_size * kernel_size, output_pooling_size});
        ShapeHandle valid_idx_shape = c->MakeShape({center_num});

        c->set_output(0, output_idx_shape); // output_idx
        c->set_output(1, valid_idx_shape); // valid_idx

        return Status::OK();

    }); // InferenceContext


void voxel_sampling_idx_gpu_launcher(int batch_size, int input_point_num,
                                     int center_num, int kernel_size,
                                     int grid_dim_w, int grid_dim_l, int grid_dim_h, std::vector<float> resolution,
                                     int grid_buffer_size, int output_pooling_size, bool with_rpn,
                                     const float* input_coors,
                                     const int* input_num_list,
                                     const float* center_coors,
                                     const int* center_num_list,
                                     int* input_accu_list,
                                     int* center_accu_list,
                                     int* grid_buffer,
                                     int* grid_buffer_count,
                                     int* output_idx,
                                     int* output_idx_count,
                                     int* valid_idx);

class VoxelSamplingIdxOp: public OpKernel {
public:
    explicit VoxelSamplingIdxOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("resolution", &resolution));
        OP_REQUIRES_OK(context, context->GetAttr("with_rpn", &with_rpn));
        OP_REQUIRES_OK(context, context->GetAttr("dimension", &dimension));
        OP_REQUIRES_OK(context, context->GetAttr("grid_buffer_size", &grid_buffer_size));
        OP_REQUIRES_OK(context, context->GetAttr("output_pooling_size", &output_pooling_size));
        OP_REQUIRES(context, resolution.size() == 3 && resolution[0] * resolution[1] * resolution[2] > 0,
                    errors::InvalidArgument("Resolution has to be in 3-D and greater than 0"));
        OP_REQUIRES(context, dimension.size() == 3,
                    errors::InvalidArgument("Dimension has to be 3-D for Voxel Sample Operation."));
    }
    void Compute(OpKernelContext* context) override {

        const Tensor& input_coors = context->input(0);
        auto input_coors_ptr = input_coors.template flat<float>().data();
        OP_REQUIRES(context, input_coors.dims()==2 && input_coors.dim_size(1)==3,
                    errors::InvalidArgument("Voxel Sample Op expects input_coors in shape: [point_nums, 3]."));

        const Tensor& input_num_list = context->input(1);
        auto input_num_list_ptr = input_num_list.template flat<int>().data();
        OP_REQUIRES(context, input_num_list.dims()==1,
                    errors::InvalidArgument("Voxel Sample Op expects input_num_list in shape: [batch_size]."));

        const Tensor& center_coors = context->input(2);
        auto center_coors_ptr = center_coors.template flat<float>().data();
        OP_REQUIRES(context, center_coors.dims()==2 && center_coors.dim_size(1)==3,
                    errors::InvalidArgument("Voxel Sample Op expects center coors in shape: [ncenters, 3]."));

        const Tensor& center_num_list = context->input(3);
        auto center_num_list_ptr = center_num_list.template flat<int>().data();
        OP_REQUIRES(context, center_num_list.dims()==1,
                    errors::InvalidArgument("Voxel Sample Op expects center_num_list in shape: [batch_size]."));

        int kernel_size = 3;
        int input_point_num = input_coors.dim_size(0);
        int center_num = center_coors.dim_size(0);
        int batch_size = input_num_list.dim_size(0);
        int kernel_num = kernel_size * kernel_size * kernel_size;
        int grid_dim_w = (int)floor(dimension[0] / resolution[0]);
        int grid_dim_l = (int)floor(dimension[1] / resolution[1]);
        int grid_dim_h = (int)floor(dimension[2] / resolution[2]);
        if (INT_MAX / grid_dim_h / grid_dim_l / grid_dim_w < batch_size){
            printf("VoxelSamplingOp ERROR: size of grid buffer %d x [%d x %d x %d] exceeds INT32 range: %d.\n",
	                batch_size, grid_dim_w, grid_dim_l, grid_dim_h, INT_MAX);}

//        printf("******************input shape = %d************************\n", input_point_num);
//        printf("******************output shape = %d************************\n", kernel_number);



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
        }
//
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

        Tensor grid_buffer;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{batch_size, grid_dim_w, grid_dim_l, grid_dim_h, grid_buffer_size},
                                                       &grid_buffer));
        int* grid_buffer_ptr = grid_buffer.template flat<int>().data();
        cudaMemset(grid_buffer_ptr, 0xEF, batch_size*grid_dim_w*grid_dim_l*grid_dim_h*grid_buffer_size*sizeof(int));

        Tensor grid_buffer_count;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{batch_size, grid_dim_w, grid_dim_l, grid_dim_h},
                                                       &grid_buffer_count));
        int* grid_buffer_count_ptr = grid_buffer_count.template flat<int>().data();
        cudaMemset(grid_buffer_count_ptr, 0, batch_size*grid_dim_w*grid_dim_l*grid_dim_h*sizeof(int));

        Tensor* output_idx = nullptr;
        auto output_idx_shape = TensorShape({center_num, kernel_num, output_pooling_size});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_idx_shape, &output_idx));
        int* output_idx_ptr = output_idx->template flat<int>().data();

        Tensor output_idx_count;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{center_num, kernel_num},
                                                       &output_idx_count));
        int* output_idx_count_ptr = output_idx_count.template flat<int>().data();
        cudaMemset(output_idx_count_ptr, 0, center_num*kernel_num*sizeof(int));

        Tensor* valid_idx = nullptr;
        auto valid_idx_shape = TensorShape({center_num});
        OP_REQUIRES_OK(context, context->allocate_output(1, valid_idx_shape, &valid_idx));
        int* valid_idx_ptr = valid_idx->template flat<int>().data();
        cudaMemset(valid_idx_ptr, 0, center_num*sizeof(int));

        voxel_sampling_idx_gpu_launcher(batch_size, input_point_num,
                                        center_num, kernel_size,
                                        grid_dim_w, grid_dim_l, grid_dim_h, resolution,
                                        grid_buffer_size, output_pooling_size, with_rpn,
                                        input_coors_ptr,
                                        input_num_list_ptr,
                                        center_coors_ptr,
                                        center_num_list_ptr,
                                        input_accu_list_ptr,
                                        center_accu_list_ptr,
                                        grid_buffer_ptr,
                                        grid_buffer_count_ptr,
                                        output_idx_ptr,
                                        output_idx_count_ptr,
                                        valid_idx_ptr);

        free(input_num_list_ptr_host);
        free(center_num_list_ptr_host);
    }
private:
    bool with_rpn;
    int output_pooling_size, grid_buffer_size;
    std::vector<float> dimension;
    std::vector<float> resolution;
}; // OpKernel
REGISTER_KERNEL_BUILDER(Name("VoxelSamplingIdxOp").Device(DEVICE_GPU), VoxelSamplingIdxOp);
