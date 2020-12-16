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

REGISTER_OP("RoiFilterOp")
    .Input("input_roi_conf: float32")
    .Input("input_num_list: int32")
    .Output("output_num_list: int32") // [center_coors.shape[0], voxel_size ** 3, channels]
    .Output("output_idx: int32") // [center_coors.shape[0], voxel_size ** 3, channels]
    .Attr("conf_thres: float")
    .SetShapeFn([](InferenceContext* c){
        ShapeHandle input_roi_conf_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_roi_conf_shape));
        ShapeHandle input_num_list_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input_num_list_shape));

        DimensionHandle input_npoint = c->Dim(input_roi_conf_shape, 0);
        DimensionHandle batch_size = c->Dim(input_num_list_shape, 1);

        // The output shape during the shape inference stage is pseudo.
        ShapeHandle output_num_list_shape = c->MakeShape({batch_size});
        ShapeHandle output_idx_shape = c->MakeShape({input_npoint});

        c->set_output(0, output_num_list_shape); // output_features
        c->set_output(1, output_idx_shape); // output_idx

        return Status::OK();
    }); // InferenceContext


class RoiFilterOp: public OpKernel {
public:
    explicit RoiFilterOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("conf_thres", &conf_thres));
    }
    void Compute(OpKernelContext* context) override {

        const Tensor& input_roi_conf = context->input(0);
        auto input_roi_conf_ptr = input_roi_conf.template flat<float>().data();
        OP_REQUIRES(context, input_roi_conf.dims()==1,
                    errors::InvalidArgument("RoIFilterOp expects input_roi_conf in shape: [npoints]."));

        const Tensor& input_num_list = context->input(1);
        auto input_num_list_ptr = input_num_list.template flat<int>().data();
        OP_REQUIRES(context, input_num_list.dims()==1,
                    errors::InvalidArgument("RoIFilterOp expects input_num_list in shape: [batch_size]."));
//
        int input_npoint = input_roi_conf.dim_size(0);
        int batch_size = input_num_list.dim_size(0);
        std::vector<int> output_idx_list;

        int* input_accu_list_ptr = (int*)malloc(batch_size * sizeof(int));
        input_accu_list_ptr[0] = 0;
        for (int i=1; i<batch_size; i++)
            input_accu_list_ptr[i] = input_accu_list_ptr[i-1] + input_num_list_ptr[i-1];

        Tensor* output_num_list = nullptr;
        auto output_num_list_shape = TensorShape({batch_size});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_num_list_shape, &output_num_list));
        int* output_num_list_ptr = output_num_list->template flat<int>().data();
        memset(output_num_list_ptr, 0, batch_size * sizeof(int));

        for (int b=0; b<batch_size; b++) {
            for (int i=0; i<input_num_list_ptr[b]; i++) {
                int id = input_accu_list_ptr[b] + i;
                if (input_roi_conf_ptr[id] >= conf_thres) {
                    output_num_list_ptr[b] += 1;
                    output_idx_list.push_back(id);
                }
            }
        }
        free(input_accu_list_ptr);

        int output_npoint = output_idx_list.size();
        Tensor* output_idx = nullptr;
        auto output_idx_shape = TensorShape({output_npoint});
        OP_REQUIRES_OK(context, context->allocate_output(1, output_idx_shape, &output_idx));
        int* output_idx_ptr = output_idx->template flat<int>().data();
        for (int i=0; i<output_npoint; i++) {
            output_idx_ptr[i] = output_idx_list[i];
        }
    }
private:
    float conf_thres;
}; // OpKernel
REGISTER_KERNEL_BUILDER(Name("RoiFilterOp").Device(DEVICE_CPU), RoiFilterOp);

