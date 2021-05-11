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

REGISTER_OP("BevOccupyOp")
    .Input("input_coors: float32")
    .Input("input_num_list: int32")
    .Output("output_occupy: int32") // [center_coors.shape[0], kernel_size ** 3, channels]
    .Attr("dimension: list(float)")
    .Attr("resolution: float")
    .SetShapeFn([](InferenceContext* c){
        ShapeHandle input_num_list_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input_num_list_shape));

        DimensionHandle batch_size = c->Dim(input_num_list_shape, 0);

        int buffer_size;
        float resolution;
        std::vector<float> dimension;
        TF_RETURN_IF_ERROR(c->GetAttr("resolution", &resolution));
        TF_RETURN_IF_ERROR(c->GetAttr("dimension", &dimension));
        int output_w = (int)ceil(dimension[0] / resolution);
        int output_l = (int)ceil(dimension[1] / resolution);

        // The output shape during the shape inference stage is pseudo.
        ShapeHandle output_occupy_shape = c->MakeShape({batch_size, output_w, output_l, 1});

        c->set_output(0, output_occupy_shape); // output_idx

        return Status::OK();

    }); // InferenceContext


void bev_occupy_gpu_launcher(int batch_size, int input_point_num,
                             int output_w, int output_l, float resolution,
                             const float* input_coors,
                             const int* input_num_list,
                             int* input_accu_list,
                             int* output_occupy);

class BevOccupyOp: public OpKernel {
public:
    explicit BevOccupyOp(OpKernelConstruction* context): OpKernel(context) {
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
                    errors::InvalidArgument("BevProjectionOp expects input_coors in shape: [point_nums, 3]."));

        const Tensor& input_num_list = context->input(1);
        auto input_num_list_ptr = input_num_list.template flat<int>().data();
        OP_REQUIRES(context, input_num_list.dims()==1,
                    errors::InvalidArgument("BevProjectionOp expects input_num_list in shape: [batch_size]."));

        int batch_size = input_num_list.dim_size(0);
        int input_point_num = input_coors.dim_size(0);
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

        Tensor* output_occupy = nullptr;
        auto output_occupy_shape = TensorShape({batch_size, output_w, output_l, 1});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_occupy_shape, &output_occupy));
        int* output_occupy_ptr = output_occupy->template flat<int>().data();
        cudaMemset(output_occupy_ptr, 0, batch_size*output_w*output_l*1*sizeof(int));

        bev_occupy_gpu_launcher(batch_size, input_point_num,
                                output_w, output_l, resolution,
                                input_coors_ptr,
                                input_num_list_ptr,
                                input_accu_list_ptr,
                                output_occupy_ptr);

        free(input_num_list_ptr_host);
    }
private:
    float resolution;
    std::vector<float> dimension;
}; // OpKernel
REGISTER_KERNEL_BUILDER(Name("BevOccupyOp").Device(DEVICE_GPU), BevOccupyOp);

