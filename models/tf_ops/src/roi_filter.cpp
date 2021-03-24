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
#include <random>
#include <chrono>
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
    .Input("input_roi_ious: float32")
    .Input("input_num_list: int32")
    .Output("output_num_list: int32")
    .Output("output_idx: int32")
    .Attr("conf_thres: float")
    .Attr("iou_thres: float")
    .Attr("max_length: int")
    .Attr("with_negative: bool")
    .SetShapeFn([](InferenceContext* c){
        ShapeHandle input_roi_conf_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_roi_conf_shape));
        ShapeHandle input_num_list_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input_num_list_shape));

        DimensionHandle input_point_num = c->Dim(input_roi_conf_shape, 0);
        DimensionHandle batch_size = c->Dim(input_num_list_shape, 0);

        // The output shape during the shape inference stage is pseudo.
        ShapeHandle output_num_list_shape = c->MakeShape({batch_size});
        ShapeHandle output_idx_shape = c->MakeShape({input_point_num});

        c->set_output(0, output_num_list_shape); // output_features
        c->set_output(1, output_idx_shape); // output_idx

        return Status::OK();
    }); // InferenceContext


class RoiFilterOp: public OpKernel {
public:
    explicit RoiFilterOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("conf_thres", &conf_thres));
        OP_REQUIRES_OK(context, context->GetAttr("iou_thres", &iou_thres));
        OP_REQUIRES_OK(context, context->GetAttr("max_length", &max_length));
        OP_REQUIRES_OK(context, context->GetAttr("with_negative", &with_negative));
    }
    void Compute(OpKernelContext* context) override {

        const Tensor& input_roi_conf = context->input(0);
        auto input_roi_conf_ptr = input_roi_conf.template flat<float>().data();
        OP_REQUIRES(context, input_roi_conf.dims()==1,
                    errors::InvalidArgument("RoIFilterOp expects input_roi_conf in shape: [npoints]."));

        const Tensor& input_roi_ious = context->input(1);
        auto input_roi_ious_ptr = input_roi_ious.template flat<float>().data();
        OP_REQUIRES(context, input_roi_ious.dims()==1,
                    errors::InvalidArgument("RoIFilterOp expects input_roi_ious in shape: [npoints]."));

        const Tensor& input_num_list = context->input(2);
        auto input_num_list_ptr = input_num_list.template flat<int>().data();
        OP_REQUIRES(context, input_num_list.dims()==1,
                    errors::InvalidArgument("RoIFilterOp expects input_num_list in shape: [batch_size]."));
//
        int input_point_num = input_roi_conf.dim_size(0);
        int batch_size = input_num_list.dim_size(0);
        max_length = (max_length > 0) ? max_length : 500;
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
            int positive_count = 0;
            int negative_count = 0;
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::srand(seed);
            std::vector<int> id_list(input_num_list_ptr[b]);
            for (int i=0; i<input_num_list_ptr[b]; i++)
                id_list[i] = i;
            std::random_shuffle(id_list.begin(), id_list.end());


            if (with_negative) {
                for (int i=0; i<input_num_list_ptr[b]; i++) {
                    int id = input_accu_list_ptr[b] + id_list[i];
                    if (input_roi_conf_ptr[id] >= conf_thres &&
                        positive_count < max_length / 2 &&
                        input_roi_ious_ptr[id] >= iou_thres) {
                        output_num_list_ptr[b] += 1;
                        positive_count += 1;
                        output_idx_list.push_back(id);
                    }
                }

                for (int i=0; i<input_num_list_ptr[b]; i++) {
                    int id = input_accu_list_ptr[b] + id_list[i];
                    if (input_roi_conf_ptr[id] >= conf_thres &&
                        negative_count < max_length / 2 &&
                        input_roi_ious_ptr[id] < iou_thres &&
                        negative_count < positive_count) {
                        output_num_list_ptr[b] += 1;
                        negative_count += 1;
                        output_idx_list.push_back(id);
                    }
                }
//                printf("Positive count=%d, negative count=%d\n", positive_count, negative_count);
            } else {
                for (int i=0; i<input_num_list_ptr[b]; i++) {
                    int id = input_accu_list_ptr[b] + id_list[i];
                    if (input_roi_conf_ptr[id] >= conf_thres) {
                        output_num_list_ptr[b] += 1;
                        positive_count += 1;
                        output_idx_list.push_back(id);
                    }
                }
            }



//            for (int i=0; i<input_num_list_ptr[b]; i++) {
//                int id = input_accu_list_ptr[b] + id_list[i];
//                if (with_negative) {
//                    if (input_roi_conf_ptr[id] >= conf_thres &&
//                        positive_count < max_length / 2 &&
//                        input_roi_ious_ptr[id] >= iou_thres) {
//                        output_num_list_ptr[b] += 1;
//                        positive_count += 1;
//                        output_idx_list.push_back(id);
//                    }
//                    if (input_roi_conf_ptr[id] >= conf_thres &&
//                        negative_count < max_length / 2 &&
//                        input_roi_ious_ptr[id] < iou_thres) {
//                        output_num_list_ptr[b] += 1;
//                        negative_count += 1;
//                        output_idx_list.push_back(id);
//                    }
//                } else {
//                    if (input_roi_conf_ptr[id] >= conf_thres && positive_count < max_length) {
//                        output_num_list_ptr[b] += 1;
//                        positive_count += 1;
//                        output_idx_list.push_back(id);
//                    }
//                }
//            }




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
    float conf_thres, iou_thres;
    int max_length;
    bool with_negative;
}; // OpKernel
REGISTER_KERNEL_BUILDER(Name("RoiFilterOp").Device(DEVICE_CPU), RoiFilterOp);

