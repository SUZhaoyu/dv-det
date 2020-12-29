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

REGISTER_OP("BboxLogitsToAttrsOp")
    .Input("input_roi_attrs: float32")
    .Input("input_logits: float32")
    .Output("output_attrs: float32")
    .SetShapeFn([](InferenceContext* c){
        ShapeHandle input_roi_attrs_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_roi_attrs_shape));

        DimensionHandle input_npoint = c->Dim(input_roi_attrs_shape, 0);
        ShapeHandle output_attrs_shape = c->MakeShape({input_npoint, 7});
        c->set_output(0, output_attrs_shape);

        return Status::OK();

    }); // InferenceContext

void bbox_logits_to_attrs_gpu_launcher(int input_npoint, int channels,
                                       const float* input_roi_attrs,
                                       const float* input_logits,
                                       float* output_attrs);

class BboxLogitsToAttrsOp: public OpKernel {
public:
    explicit BboxLogitsToAttrsOp(OpKernelConstruction* context): OpKernel(context) {}
    void Compute(OpKernelContext* context) override {

        const Tensor& input_roi_attrs = context->input(0);
        auto input_roi_attrs_ptr = input_roi_attrs.template flat<float>().data();
        OP_REQUIRES(context, input_roi_attrs.dims()==2 && input_roi_attrs.dim_size(1)==7,
                    errors::InvalidArgument("RoiLogitsToAttrsOp expects input_roi_attrs in shape: [npoints, 7]."));

        const Tensor& input_logits = context->input(1);
        auto input_logits_ptr = input_logits.template flat<float>().data();
        OP_REQUIRES(context, input_logits.dims()==2 && input_logits.dim_size(1)>=7,
                    errors::InvalidArgument("RoiLogitsToAttrsOp expects input_logits in shape: [npoints, logits(>=7)]."));
        OP_REQUIRES(context, input_logits.dim_size(0) == input_roi_attrs.dim_size(0),
                    errors::InvalidArgument("RoiLogitsToAttrsOp expects input_roi_attrs and input_logits has the same length."));

        int input_npoint = input_roi_attrs.dim_size(0);
        int channels = input_logits.dim_size(1);

        Tensor* output_attrs = nullptr;
        auto output_attrs_shape = TensorShape({input_npoint, 7});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_attrs_shape, &output_attrs));
        float* output_attrs_ptr = output_attrs->template flat<float>().data();
        cudaMemset(output_attrs_ptr, 0, input_npoint*7*sizeof(int));

        bbox_logits_to_attrs_gpu_launcher(input_npoint, channels,
                                          input_roi_attrs_ptr,
                                          input_logits_ptr,
                                          output_attrs_ptr);
    }
}; // OpKernel
REGISTER_KERNEL_BUILDER(Name("BboxLogitsToAttrsOp").Device(DEVICE_GPU), BboxLogitsToAttrsOp);