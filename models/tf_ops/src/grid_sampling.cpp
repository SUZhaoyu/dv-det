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
#include <climits>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("GridSamplingOp")
    .Input("input_coors: float32")
    .Input("input_num_list: int32")
    .Output("output_idx: int32")
    .Output("output_num_list: int32")
    .Attr("dimension: list(float)")
    .Attr("resolution: float")
    .SetShapeFn([](InferenceContext* c){
        ShapeHandle input_coors_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_coors_shape));
        ShapeHandle input_num_list_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input_num_list_shape));

        DimensionHandle input_npoint = c->Dim(input_coors_shape, 0);
        ShapeHandle output_idx_shape = c->MakeShape({input_npoint});
        c->set_output(0, output_idx_shape);
        c->set_output(1, input_num_list_shape);

        return Status::OK();

    }); // InferenceContext

void grid_sampling_gpu_launcher(int batch_size, int input_npoint, float resolution,
                                int grid_w, int grid_l, int grid_h,
                                const float* input_coors,
                                const int* input_num_list,
                                int* input_accu_list,
                                int* output_idx_temp,
                                int* output_num_list,
                                int* grid_buffer);

class GridSamplingOp: public OpKernel {
public:
    explicit GridSamplingOp(OpKernelConstruction* context): OpKernel(context) {
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
        OP_REQUIRES(context, input_coors.dims()==2,
                    errors::InvalidArgument("GridSamplingOp expects input_coors in 2-D"));

        const Tensor& input_num_list = context->input(1);
        auto input_num_list_ptr = input_num_list.template flat<int>().data();
        OP_REQUIRES(context, input_num_list.dims()==1,
                    errors::InvalidArgument("GridSamplingOp expects input_num_list in 1-D."));

        int input_npoint = input_coors.dim_size(0);
        int batch_size = input_num_list.dim_size(0);
        int grid_w = (int)floor(dimension[0] / resolution);
        int grid_l = (int)floor(dimension[1] / resolution);
        int grid_h = (int)floor(dimension[2] / resolution);
        if (INT_MAX / grid_h / grid_l / grid_w < batch_size){
            printf("GridSamplingOp ERROR: size of grid buffer %d x [%d x %d x %d] exceeds INT32 range: %d.\n",
	                batch_size, grid_w, grid_l, grid_h, INT_MAX);}


        int* input_num_list_ptr_host = (int*)malloc(batch_size*sizeof(int));
        cudaMemcpy(input_num_list_ptr_host, input_num_list_ptr, batch_size*sizeof(int), cudaMemcpyDeviceToHost);
        int* input_accu_list_ptr_host = (int*)malloc(batch_size*sizeof(int));
        input_accu_list_ptr_host[0] = 0;
        for (int i=1; i<batch_size; i++)
            input_accu_list_ptr_host[i] = input_accu_list_ptr_host[i-1] + input_num_list_ptr_host[i-1];

        Tensor input_accu_list;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{batch_size},
                                                       &input_accu_list));
        int* input_accu_list_ptr = input_accu_list.template flat<int>().data();
        cudaMemcpy(input_accu_list_ptr, input_accu_list_ptr_host, batch_size*sizeof(int), cudaMemcpyHostToDevice);

        Tensor output_idx_temp;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{input_npoint},
                                                       &output_idx_temp));
        int* output_idx_temp_ptr = output_idx_temp.template flat<int>().data();
        cudaMemset(output_idx_temp_ptr, 0, input_npoint*sizeof(int));

        Tensor grid_buffer;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{batch_size, grid_w, grid_l, grid_h},
                                                       &grid_buffer));
        int* grid_buffer_ptr = grid_buffer.template flat<int>().data();
        cudaMemset(grid_buffer_ptr, 0, batch_size*grid_w*grid_l*grid_h*sizeof(int));


        Tensor* output_num_list = nullptr;
        auto output_num_list_shape = TensorShape({batch_size});
        OP_REQUIRES_OK(context, context->allocate_output(1, output_num_list_shape, &output_num_list));
        int* output_num_list_ptr = output_num_list->template flat<int>().data();
        cudaMemset(output_num_list_ptr, 0, batch_size*sizeof(int));

        grid_sampling_gpu_launcher(batch_size, input_npoint, resolution,
                                   grid_w, grid_l, grid_h,
                                   input_coors_ptr,
                                   input_num_list_ptr,
                                   input_accu_list_ptr,
                                   output_idx_temp_ptr,
                                   output_num_list_ptr,
                                   grid_buffer_ptr);

        int* output_idx_temp_ptr_host = (int*)malloc(input_npoint*sizeof(int));
        cudaMemcpy(output_idx_temp_ptr_host, output_idx_temp_ptr, input_npoint*sizeof(int), cudaMemcpyDeviceToHost);
        int* output_num_list_ptr_host = (int*)malloc(batch_size*sizeof(int));
        cudaMemcpy(output_num_list_ptr_host, output_num_list_ptr, batch_size*sizeof(int), cudaMemcpyDeviceToHost);


        int output_count = 0;
        for (int i=0; i<batch_size; i++) {
            output_count += output_num_list_ptr_host[i];
        }
//        printf("******************input shape = %d************************\n", input_npoint);
//        printf("******************output shape = %d************************\n", output_count);
        Tensor* output_idx = nullptr;
        auto output_idx_shape = TensorShape({output_count});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_idx_shape, &output_idx));
        int* output_idx_ptr = output_idx->template flat<int>().data();
        int target_count = 0;
        int source_count = 0;
        for (int i=0; i<batch_size; i++) {
            int cpy_size = output_num_list_ptr_host[i] * sizeof(int);
            cudaMemcpy(&output_idx_ptr[target_count], &output_idx_temp_ptr_host[source_count], cpy_size, cudaMemcpyHostToDevice);
//            for (int j=0; j<output_num_list_ptr_host[i]; j++) {
//                if (output_idx_temp_ptr_host[source_count + j] < source_count || output_idx_temp_ptr_host[source_count + j] >= source_count + input_num_list_ptr_host[i])
//                printf("Batch-%d point Id %d exceed range [%d, %d]\n", i, output_idx_temp_ptr_host[source_count + j], source_count, source_count + input_num_list_ptr_host[i]);
//                printf("Batch-%d point Id[%d] = %d.\n", i, j, output_idx_temp_ptr_host[source_count + j]);
//            }
            target_count += output_num_list_ptr_host[i];
            source_count += input_num_list_ptr_host[i];

        }
        free(output_idx_temp_ptr_host);
        free(output_num_list_ptr_host);
        free(input_accu_list_ptr_host);
        cudaFree(grid_buffer_ptr);
        cudaFree(input_accu_list_ptr);
    }
private:
    float resolution;
    std::vector<float> dimension;
};
REGISTER_KERNEL_BUILDER(Name("GridSamplingOp").Device(DEVICE_GPU), GridSamplingOp);
