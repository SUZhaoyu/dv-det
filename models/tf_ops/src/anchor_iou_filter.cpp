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

REGISTER_OP("AnchorIouFilterOp")
    .Input("anchor_iou: float32")
    .Input("iou_idx: int32")
    .Input("label_bbox: float32")
    .Input("gt_conf: int32")
    .Input("valid_idx: int32" )
    .Output("iou_mask: int32")
    .Attr("thres_high: float")
    .Attr("thres_low: float")
    .SetShapeFn([](InferenceContext* c){
        c->set_output(0, c->input(3));
        return Status::OK();
    }); // InferenceContext


class AnchorIouFilterOp: public OpKernel {
public:
    explicit AnchorIouFilterOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("thres_high", &thres_high));
        OP_REQUIRES_OK(context, context->GetAttr("thres_low", &thres_low));
    }
    void Compute(OpKernelContext* context) override {

        const Tensor& anchor_iou = context->input(0);
        auto anchor_iou_ptr = anchor_iou.template flat<float>().data();
        OP_REQUIRES(context, anchor_iou.dims()==1,
                    errors::InvalidArgument("AnchorIouFilterOp expects anchor_iou in shape: [n_anchor]."));

        const Tensor& iou_idx = context->input(1);
        auto iou_idx_ptr = iou_idx.template flat<int>().data();
        OP_REQUIRES(context, iou_idx.dims()==1,
                    errors::InvalidArgument("AnchorIouFilterOp expects iou_idx in shape: [n_anchor]."));

        const Tensor& label_bbox = context->input(2);
        auto label_bbox_ptr = label_bbox.template flat<float>().data();
        OP_REQUIRES(context, label_bbox.dims()==3,
                    errors::InvalidArgument("AnchorIouFilterOp expects label_bbox in shape: [batch_size, nbbox, bbox_attr]."));

        const Tensor& gt_conf = context->input(3);
        auto gt_conf_ptr = gt_conf.template flat<int>().data();
        OP_REQUIRES(context, gt_conf.dims()==1,
                    errors::InvalidArgument("AnchorIouFilterOp expects gt_conf in shape: [n]."));

        const Tensor& valid_idx = context->input(4);
        auto valid_idx_ptr = valid_idx.template flat<int>().data();
        OP_REQUIRES(context, valid_idx.dims()==1,
                    errors::InvalidArgument("AnchorIouFilterOp expects valid_idx in shape: [n]."));


        int n_anchor = anchor_iou.dim_size(0);
        int batch_size = label_bbox.dim_size(0);
        int nbbox = label_bbox.dim_size(1);
        int gt_length = gt_conf.dim_size(0);

        Tensor iou_buffer;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value,
                                                       TensorShape{batch_size * nbbox},
                                                       &iou_buffer));
        float* iou_buffer_ptr = iou_buffer.template flat<float>().data();
        memset(iou_buffer_ptr, 0, batch_size * nbbox * sizeof(float));

        Tensor idx_buffer;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{batch_size * nbbox},
                                                       &idx_buffer));
        int* idx_buffer_ptr = idx_buffer.template flat<int>().data();
        memset(iou_buffer_ptr, 0, batch_size * nbbox * sizeof(int));

        Tensor iou_mask_temp;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{n_anchor},
                                                       &iou_mask_temp));
        int* iou_mask_temp_ptr = iou_mask_temp.template flat<int>().data();
        memset(iou_mask_temp_ptr, -1, n_anchor * sizeof(int));
//        for (int i=0; i<n_anchor; ++i) {
//            printf("%d\n", iou_mask_temp_ptr[i]);
//        }

        Tensor* output_gt_conf = nullptr;
        auto output_gt_conf_shape = TensorShape({gt_length});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_gt_conf_shape, &output_gt_conf));
        int* output_gt_conf_ptr = output_gt_conf->template flat<int>().data();
        memcpy(output_gt_conf_ptr, gt_conf_ptr, gt_length * sizeof(int));

        for (int i=0; i<n_anchor; ++i) {
            float iou = anchor_iou_ptr[i];
            int label_idx = iou_idx_ptr[i];
            if (iou >= thres_high)
                iou_mask_temp_ptr[i] = 1;
            if (iou <= thres_low)
                iou_mask_temp_ptr[i] = 0;
            if (iou > iou_buffer_ptr[label_idx]) {
                iou_buffer_ptr[label_idx] = iou;
                idx_buffer_ptr[label_idx] = i;
            }
        }
        for (int i=0; i<batch_size * nbbox; ++i) {
            if (iou_buffer_ptr[i] > 0.05) {
                iou_mask_temp_ptr[idx_buffer_ptr[i]] = 1;
            }
        }
        for (int i=0; i<n_anchor; ++i) {
//            printf("%d -> %d\n", output_gt_conf_ptr[valid_idx_ptr[i]], iou_mask_temp_ptr[i]);
            output_gt_conf_ptr[valid_idx_ptr[i]] = iou_mask_temp_ptr[i];
        }
    }
private:
    float thres_high, thres_low;
}; // OpKernel
REGISTER_KERNEL_BUILDER(Name("AnchorIouFilterOp").Device(DEVICE_CPU), AnchorIouFilterOp);

