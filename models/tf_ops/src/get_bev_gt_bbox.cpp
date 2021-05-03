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

REGISTER_OP("GetBevGtBboxOp")
    .Input("input_coors: float32")
    .Input("label_bbox: float32")
    .Input("input_num_list: int32")
    .Input("anchor_param_list: float32")
    .Output("gt_bbox: float32")
    .Output("gt_conf: int32")
    .Output("label_idx: int32")
    .Attr("expand_ratio: float")
    .Attr("diff_thres: int")
    .Attr("cls_thres: int")
    .SetShapeFn([](InferenceContext* c){
        ShapeHandle input_coors_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_coors_shape));

        ShapeHandle anchor_param_list_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &anchor_param_list_shape));

        DimensionHandle npoint = c->Dim(input_coors_shape, 0);
        DimensionHandle num_anchor = c->Dim(anchor_param_list_shape, 0);
        ShapeHandle gt_bbox_shape = c->MakeShape({npoint, num_anchor, 7});
        ShapeHandle gt_conf_shape = c->MakeShape({npoint, num_anchor});
        ShapeHandle label_idx_shape = c->MakeShape({npoint, num_anchor});
        c->set_output(0, gt_bbox_shape);
        c->set_output(1, gt_conf_shape);
        c->set_output(2, label_idx_shape);

        return Status::OK();

    }); // InferenceContext

void get_bev_gt_bbox_gpu_launcher(int batch_size, int npoint, int nbbox, int bbox_attr,
                                  int num_anchor, int anchor_attr,
                                  int diff_thres, int cls_thres, float expand_ratio,
                                  const float* input_coors,
                                  const float* label_bbox,
                                  const int* input_num_list,
                                  const float* anchor_param_list,
                                  int* input_accu_list,
                                  float* gt_bbox,
                                  int* gt_conf,
                                  int* label_idx);

class GetBevGtBboxOp: public OpKernel {
public:
    explicit GetBevGtBboxOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("diff_thres", &diff_thres));
        OP_REQUIRES_OK(context, context->GetAttr("cls_thres", &cls_thres));
        OP_REQUIRES_OK(context, context->GetAttr("expand_ratio", &expand_ratio));
    }
    void Compute(OpKernelContext* context) override {
        const Tensor& input_coors = context->input(0);
        auto input_coors_ptr = input_coors.template flat<float>().data();
        OP_REQUIRES(context, input_coors.dim_size(1) == 2,
            errors::InvalidArgument("The attribute of bev lidar coors has to be 2."));

        const Tensor& label_bbox = context->input(1);
        auto label_bbox_ptr = label_bbox.template flat<float>().data();
        OP_REQUIRES(context, label_bbox.dim_size(2)==9,
                    errors::InvalidArgument("Attribute of bbox has to be 9: [l, h, w, x, y, z, r, cls, diff_idx]."));

        const Tensor& input_num_list = context->input(2);
        auto input_num_list_ptr = input_num_list.template flat<int>().data();
        OP_REQUIRES(context, input_num_list.dims()==1,
                    errors::InvalidArgument("GetGtBboxOp Op expects input in shape: [batch_size]."));

        const Tensor& anchor_param_list = context->input(3);
        auto anchor_param_list_ptr = anchor_param_list.template flat<float>().data();
        OP_REQUIRES(context, anchor_param_list.dims()==2 && anchor_param_list.dim_size(1)==5,
                    errors::InvalidArgument("GetGtBboxOp Op expects anchor_param_list in shape: [n_anchor, 5]."));

        int batch_size = input_num_list.dim_size(0);
        int bbox_attr = label_bbox.dim_size(2);
        int npoint = input_coors.dim_size(0);
        int nbbox = label_bbox.dim_size(1);
        int num_anchor = anchor_param_list.dim_size(0);
        int anchor_attr = anchor_param_list.dim_size(1);

        Tensor input_accu_list;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{batch_size},
                                                       &input_accu_list));
        int* input_accu_list_ptr = input_accu_list.template flat<int>().data();
        cudaMemset(input_accu_list_ptr, 0, batch_size * sizeof(int));

        Tensor* gt_bbox = nullptr;
        auto gt_bbox_shape = TensorShape({npoint, num_anchor, 7});
        OP_REQUIRES_OK(context, context->allocate_output(0, gt_bbox_shape, &gt_bbox));
        float* gt_bbox_ptr = gt_bbox->template flat<float>().data();
        cudaMemset(gt_bbox_ptr, 0.f, npoint * num_anchor * 7 * sizeof(float));

        Tensor* gt_conf = nullptr;
        auto gt_conf_shape = TensorShape({npoint, num_anchor});
        OP_REQUIRES_OK(context, context->allocate_output(1, gt_conf_shape, &gt_conf));
        int* gt_conf_ptr = gt_conf->template flat<int>().data();
        cudaMemset(gt_conf_ptr, 0, npoint * num_anchor * sizeof(int));

        Tensor* label_idx = nullptr;
        auto label_idx_shape = TensorShape({npoint, num_anchor});
        OP_REQUIRES_OK(context, context->allocate_output(2, label_idx_shape, &label_idx));
        int* label_idx_ptr = label_idx->template flat<int>().data();
        cudaMemset(label_idx_ptr, 0, npoint * num_anchor * sizeof(int));

        get_bev_gt_bbox_gpu_launcher(batch_size, npoint, nbbox, bbox_attr,
                                     num_anchor, anchor_attr,
                                     diff_thres, cls_thres, expand_ratio,
                                     input_coors_ptr,
                                     label_bbox_ptr,
                                     input_num_list_ptr,
                                     anchor_param_list_ptr,
                                     input_accu_list_ptr,
                                     gt_bbox_ptr,
                                     gt_conf_ptr,
                                     label_idx_ptr);

    }
private:
    int diff_thres, cls_thres;
    float expand_ratio;
};
REGISTER_KERNEL_BUILDER(Name("GetBevGtBboxOp").Device(DEVICE_GPU), GetBevGtBboxOp);