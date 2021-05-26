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

REGISTER_OP("ArgSortOp")
    .Input("input: float32")
    .Output("output_idx: int32")
    .Attr("descending: bool")
    .SetShapeFn([](InferenceContext* c){
        c->set_output(0, c->input(0));
        return Status::OK();
    }); // InferenceContext

template <typename T>
std::vector<int> argsort(const std::vector<T> &v, bool descending)
{
    // 建立下标数组
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    // 调用sort函数，匿名函数自动捕获待排序数组
    if (descending)
        sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v[i1] > v[i2];});
    if (!descending)
        sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v[i1] < v[i2];});
    return idx;
}



class ArgSortOp: public OpKernel {
public:
    explicit ArgSortOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("descending", &descending));
    }
    void Compute(OpKernelContext* context) override {

        const Tensor& input = context->input(0);
        auto input_ptr = input.template flat<float>().data();
        OP_REQUIRES(context, input.dims()==1,
                    errors::InvalidArgument("ArgSortOp expects input in shape: [npoint]."));

        int npoint = input.dim_size(0);
        const std::vector<float> input_vector(&input_ptr[0], &input_ptr[0]+npoint);
        std::vector<int> output_idx_vector;
        output_idx_vector = argsort(input_vector, descending);

        Tensor* output_idx = nullptr;
        auto output_idx_shape = TensorShape({npoint});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_idx_shape, &output_idx));
        int* output_idx_ptr = output_idx->template flat<int>().data();
        for (int i=0; i<npoint; i++) {
            output_idx_ptr[i] = output_idx_vector[i];
        }
    }
private:
    bool descending;
}; // OpKernel
REGISTER_KERNEL_BUILDER(Name("ArgSortOp").Device(DEVICE_CPU), ArgSortOp);

