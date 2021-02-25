/* get ground truth tensorflow cpu wrapper
 * By Zhaoyu SU, Email: zsuad@connect.ust.hk
 * All Rights Reserved. Sep, 2019.
 */
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("GetRoiBboxOp")
    .Input("input_coors: float32")
    .Input("gt_bbox: float32")
    .Input("input_num_list: int32")
    .Input("anchor_size: float32")
    .Output("roi_bbox: float32")
    .Output("roi_conf: int32")
    .Output("roi_diff: int32")
    .Attr("expand_ratio: float")
    .Attr("diff_thres: int")
    .Attr("cls_thres: int")
    .SetShapeFn([](InferenceContext* c){
        ShapeHandle input_coors_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_coors_shape));

        DimensionHandle npoint = c->Dim(input_coors_shape, 0);
        ShapeHandle roi_bbox_shape = c->MakeShape({npoint, 7});
        ShapeHandle roi_conf_shape = c->MakeShape({npoint});
        ShapeHandle roi_diff_shape = c->MakeShape({npoint});
        c->set_output(0, roi_bbox_shape);
        c->set_output(1, roi_conf_shape);
        c->set_output(2, roi_diff_shape);

        return Status::OK();

    }); // InferenceContext

void get_roi_bbox_gpu_launcher(int batch_size, int npoint, int nbbox, int bbox_attr,
                               int diff_thres, int cls_thres, float expand_ratio,
                               const float* input_coors,
                               const float* gt_bbox,
                               const int* input_num_list,
                               const float* anchor_size,
                               int* input_accu_list,
                               float* roi_bbox,
                               int* roi_conf,
                               int* roi_diff);

class GetRoiBboxOp: public OpKernel {
public:
    explicit GetRoiBboxOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("diff_thres", &diff_thres));
        OP_REQUIRES_OK(context, context->GetAttr("cls_thres", &cls_thres));
        OP_REQUIRES_OK(context, context->GetAttr("expand_ratio", &expand_ratio));
    }
    void Compute(OpKernelContext* context) override {
        const Tensor& input_coors = context->input(0);
        auto input_coors_ptr = input_coors.template flat<float>().data();
        OP_REQUIRES(context, input_coors.dim_size(1) == 3,
            errors::InvalidArgument("The attribute of lidar coors has to be 3."));

        const Tensor& gt_bbox = context->input(1);
        auto gt_bbox_ptr = gt_bbox.template flat<float>().data();
        OP_REQUIRES(context, gt_bbox.dim_size(2)==9,
                    errors::InvalidArgument("Attribute of bbox has to be 9: [l, h, w, x, y, z, r, cls, diff_idx]."));

        const Tensor& input_num_list = context->input(2);
        auto input_num_list_ptr = input_num_list.template flat<int>().data();
        OP_REQUIRES(context, input_num_list.dims()==1,
                    errors::InvalidArgument("FPS Op expects input in shape: [batch_size]."));

        const Tensor& anchor_size = context->input(3);
        auto anchor_size_ptr = anchor_size.template flat<float>().data();
        OP_REQUIRES(context, anchor_size.dims()==1 && anchor_size.dim_size(0)==3,
                    errors::InvalidArgument("FPS Op expects anchor in shape: [3]."));

        int batch_size = input_num_list.dim_size(0);
        int bbox_attr = gt_bbox.dim_size(2);
        int npoint = input_coors.dim_size(0);
        int nbbox = gt_bbox.dim_size(1);

        Tensor input_accu_list;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{batch_size},
                                                       &input_accu_list));
        int* input_accu_list_ptr = input_accu_list.template flat<int>().data();
        cudaMemset(input_accu_list_ptr, 0, batch_size * sizeof(int));

        Tensor* roi_bbox = nullptr;
        auto roi_bbox_shape = TensorShape({npoint, 7});
        OP_REQUIRES_OK(context, context->allocate_output(0, roi_bbox_shape, &roi_bbox));
        float* roi_bbox_ptr = roi_bbox->template flat<float>().data();
        cudaMemset(roi_bbox_ptr, 0.f, npoint * 7 * sizeof(float));

        Tensor* roi_conf = nullptr;
        auto roi_conf_shape = TensorShape({npoint});
        OP_REQUIRES_OK(context, context->allocate_output(1, roi_conf_shape, &roi_conf));
        int* roi_conf_ptr = roi_conf->template flat<int>().data();
        cudaMemset(roi_conf_ptr, 0, npoint * sizeof(int));

        Tensor* roi_diff = nullptr;
        auto roi_diff_shape = TensorShape({npoint});
        OP_REQUIRES_OK(context, context->allocate_output(2, roi_diff_shape, &roi_diff));
        int* roi_diff_ptr = roi_diff->template flat<int>().data();
        cudaMemset(roi_diff_ptr, 0, npoint * sizeof(int));

        get_roi_bbox_gpu_launcher(batch_size, npoint, nbbox, bbox_attr,
                                   diff_thres, cls_thres, expand_ratio,
                                   input_coors_ptr,
                                   gt_bbox_ptr,
                                   input_num_list_ptr,
                                   anchor_size_ptr,
                                   input_accu_list_ptr,
                                   roi_bbox_ptr,
                                   roi_conf_ptr,
                                   roi_diff_ptr);

    }
private:
    int diff_thres, cls_thres;
    float expand_ratio;
};
REGISTER_KERNEL_BUILDER(Name("GetRoiBboxOp").Device(DEVICE_GPU), GetRoiBboxOp);