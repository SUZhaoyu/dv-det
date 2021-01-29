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

REGISTER_OP("RoiLogitsToAttrsOp")
    .Input("base_coors: float32")
    .Input("input_logits: float32")
    .Output("output_attrs: float32")
    .Attr("anchor_size: list(float)")
    .SetShapeFn([](InferenceContext* c){
        ShapeHandle base_coors_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &base_coors_shape));

        DimensionHandle input_npoint = c->Dim(base_coors_shape, 0);
        ShapeHandle output_attrs_shape = c->MakeShape({input_npoint, 7});
        c->set_output(0, output_attrs_shape);

        return Status::OK();

    }); // InferenceContext

void roi_logits_to_attrs_gpu_launcher(int input_npoint, int channels,
                                      float anchor_w, float anchor_l, float anchor_h,
                                      const float* base_coors,
                                      const float* input_logits,
                                      float* output_attrs);

class RoiLogitsToAttrsOp: public OpKernel {
public:
    explicit RoiLogitsToAttrsOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("anchor_size", &anchor_size));
    }
    void Compute(OpKernelContext* context) override {

        const Tensor& base_coors = context->input(0);
        auto base_coors_ptr = base_coors.template flat<float>().data();
        OP_REQUIRES(context, base_coors.dims()==2 && base_coors.dim_size(1)==3,
                    errors::InvalidArgument("RoiLogitsToAttrsOp expects base_coors in shape: [npoints, 3]."));

        const Tensor& input_logits = context->input(1);
        auto input_logits_ptr = input_logits.template flat<float>().data();
        OP_REQUIRES(context, input_logits.dims()==2 && input_logits.dim_size(1)>=7,
                    errors::InvalidArgument("RoiLogitsToAttrsOp expects input_logits in shape: [npoints, logits(>=7)]."));
        OP_REQUIRES(context, input_logits.dim_size(0) == base_coors.dim_size(0),
                    errors::InvalidArgument("RoiLogitsToAttrsOp expects base_coors and input_logits has the same length."));

        int input_npoint = base_coors.dim_size(0);
        int channels = input_logits.dim_size(1);

        Tensor* output_attrs = nullptr;
        auto output_attrs_shape = TensorShape({input_npoint, 7});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_attrs_shape, &output_attrs));
        float* output_attrs_ptr = output_attrs->template flat<float>().data();
        cudaMemset(output_attrs_ptr, 0, input_npoint*7*sizeof(int));

        roi_logits_to_attrs_gpu_launcher(input_npoint, channels,
                                         anchor_size[0], anchor_size[1], anchor_size[2],
                                         base_coors_ptr,
                                         input_logits_ptr,
                                         output_attrs_ptr);
    }
private:
    std::vector<float> anchor_size;
}; // OpKernel
REGISTER_KERNEL_BUILDER(Name("RoiLogitsToAttrsOp").Device(DEVICE_GPU), RoiLogitsToAttrsOp);