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
#include <list>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("RoiPoolingOp")
    .Input("input_coors: float32")
    .Input("input_features: float32")
    .Input("roi_attrs: float32")
    .Input("input_num_list: int32")
    .Input("rois_num_list: int32")
    .Output("output_features: float32") // [center_coors.shape[0], voxel_size ** 3, channels]
    .Output("output_idx: int32") // [center_coors.shape[0], voxel_size ** 3, channels]
    .Attr("voxel_size: int")
    .Attr("padding_value: float")
    .Attr("pooling_size: int")
    .SetShapeFn([](InferenceContext* c){
        ShapeHandle input_features_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input_features_shape));
        ShapeHandle roi_attrs_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &roi_attrs_shape));

        int voxel_size;
        TF_RETURN_IF_ERROR(c->GetAttr("voxel_size", &voxel_size));

        DimensionHandle kernel_number = c->Dim(roi_attrs_shape, 0);
        DimensionHandle channels = c->Dim(input_features_shape, 1);

        // The output shape during the shape inference stage is pseudo.
        ShapeHandle output_shape = c->MakeShape({kernel_number, voxel_size * voxel_size * voxel_size, channels});

        c->set_output(0, output_shape); // output_features
        c->set_output(1, output_shape); // output_idx

        return Status::OK();
    }); // InferenceContext


REGISTER_OP("RoiPoolingGradOp")
    .Input("input_features: float32")
    .Input("output_idx: int32")
    .Input("output_features_grad: float32")
    .Output("input_features_grad: float32")
    .SetShapeFn([](InferenceContext* c){
        c->set_output(0, c->input(0));
        return Status::OK();
    }); // InferenceContext


void roi_pooling_gpu_launcher(int batch_size, int input_npoint, int channels,
                              int kernel_number, int voxel_size, int pooling_size, float padding_value,
                              const float* input_coors,
                              const float* input_features,
                              const float* roi_attrs,
                              const int* input_num_list,
                              const int* roi_num_list,
                              int* input_accu_list,
                              int* roi_accu_list,
                              int* temp_pool,
                              int* temp_count,
                              float* output_features,
                              int* output_idx);

class RoiPoolingOp: public OpKernel {
public:
    explicit RoiPoolingOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("padding_value", &padding_value));
        OP_REQUIRES_OK(context, context->GetAttr("voxel_size", &voxel_size));
        OP_REQUIRES_OK(context, context->GetAttr("pooling_size", &pooling_size));
        OP_REQUIRES(context, voxel_size % 2 == 1,
                    errors::InvalidArgument("RoIPoolingOp expects kernel size to be an odd number."));
        OP_REQUIRES(context, pooling_size > 0,
                    errors::InvalidArgument("RoIPoolingOp expects pooling size greater than 0."));
    }
    void Compute(OpKernelContext* context) override {

        const Tensor& input_coors = context->input(0);
        auto input_coors_ptr = input_coors.template flat<float>().data();
        OP_REQUIRES(context, input_coors.dims()==2 && input_coors.shape().dim_size(1)==3,
                    errors::InvalidArgument("RoIPoolingOp expects input coors in shape: [npoints, 3]."));

        const Tensor& input_features = context->input(1);
        auto input_features_ptr = input_features.template flat<float>().data();
        OP_REQUIRES(context, input_features.dims()==2 && input_features.shape().dim_size(1) > 0,
                    errors::InvalidArgument("RoIPoolingOp expects input features in shape: [npoints, channels(>0)]."));

        const Tensor& roi_attrs = context->input(2);
        auto roi_attrs_ptr = roi_attrs.template flat<float>().data();
        OP_REQUIRES(context, roi_attrs.dims()==2 && roi_attrs.shape().dim_size(1) == 7,
                    errors::InvalidArgument("RoIPoolingOp expects center coors in shape: [nRoIs, 7]."));

        const Tensor& input_num_list = context->input(3);
        auto input_num_list_ptr = input_num_list.template flat<int>().data();
        OP_REQUIRES(context, input_num_list.dims()==1,
                    errors::InvalidArgument("RoIPoolingOp  expects input_num_list in shape: [batch_size]."));

        const Tensor& roi_num_list = context->input(4);
        auto roi_num_list_ptr = roi_num_list.template flat<int>().data();
        OP_REQUIRES(context, roi_num_list.dims()==1,
                    errors::InvalidArgument("RoIPoolingOp  expects roi_num_list in shape: [batch_size]."));

        int input_npoint = input_coors.dim_size(0);
        int kernel_number = roi_attrs.dim_size(0);
        int batch_size = input_num_list.dim_size(0);
        int channels = input_features.dim_size(1);
        int ngrid = voxel_size * voxel_size * voxel_size;


        int batch_byte_size = batch_size * sizeof(int);
        int* input_num_list_ptr_host = (int*)malloc(batch_byte_size);
        int* roi_num_list_ptr_host = (int*)malloc(batch_byte_size);
        int* input_accu_list_ptr_host = (int*)malloc(batch_byte_size);
        int* roi_accu_list_ptr_host = (int*)malloc(batch_byte_size);
        input_accu_list_ptr_host[0] = 0;
        roi_accu_list_ptr_host[0] = 0;
        cudaMemcpy(input_num_list_ptr_host, input_num_list_ptr, batch_byte_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(roi_num_list_ptr_host, roi_num_list_ptr, batch_byte_size, cudaMemcpyDeviceToHost);

        for (int b=1; b<batch_size; b++) {
            input_accu_list_ptr_host[b] = input_accu_list_ptr_host[b-1] + input_num_list_ptr_host[b-1];
            roi_accu_list_ptr_host[b] = roi_accu_list_ptr_host[b-1] + roi_num_list_ptr_host[b-1];
        }


        Tensor input_accu_list;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{batch_size},
                                                       &input_accu_list));
        int* input_accu_list_ptr = input_accu_list.template flat<int>().data();
        cudaMemcpy(input_accu_list_ptr, input_accu_list_ptr_host, batch_byte_size, cudaMemcpyHostToDevice);
//        cudaMemset(&input_accu_list_ptr, 0, batch_size * sizeof(int));

        Tensor roi_accu_list;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{batch_size},
                                                       &roi_accu_list));
        int* roi_accu_list_ptr = roi_accu_list.template flat<int>().data();
//        cudaMemset(&center_accu_list_ptr, 0, batch_size * sizeof(int));
        cudaMemcpy(roi_accu_list_ptr, roi_accu_list_ptr_host, batch_byte_size, cudaMemcpyHostToDevice);


        Tensor temp_pool;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{kernel_number, ngrid, pooling_size},
                                                       &temp_pool));
        int* temp_pool_ptr = temp_pool.template flat<int>().data();


        Tensor temp_count;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{kernel_number, ngrid},
                                                       &temp_count));
        int* temp_count_ptr = temp_count.template flat<int>().data();
        cudaMemset(temp_count_ptr, 0, kernel_number*ngrid*sizeof(int));

        Tensor* output_features = nullptr;
        auto output_features_shape = TensorShape({kernel_number, ngrid, channels});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_features_shape, &output_features));
        float* output_features_ptr = output_features->template flat<float>().data();
//        cudaMemset(output_features_ptr, padding_value, kernel_number*ngrid*channels*sizeof(float));

        Tensor* output_idx = nullptr;
        auto output_idx_shape = TensorShape({kernel_number, ngrid, channels});
        OP_REQUIRES_OK(context, context->allocate_output(1, output_idx_shape, &output_idx));
        int* output_idx_ptr = output_idx->template flat<int>().data();
//        cudaMemset(output_idx_ptr, -1, kernel_number*ngrid*channels*sizeof(int));

        roi_pooling_gpu_launcher(batch_size, input_npoint, channels,
                                 kernel_number, voxel_size, pooling_size, padding_value,
                                 input_coors_ptr,
                                 input_features_ptr,
                                 roi_attrs_ptr,
                                 input_num_list_ptr,
                                 roi_num_list_ptr,
                                 input_accu_list_ptr,
                                 roi_accu_list_ptr,
                                 temp_pool_ptr,
                                 temp_count_ptr,
                                 output_features_ptr,
                                 output_idx_ptr);
    }
private:
    float padding_value;
    int voxel_size, pooling_size;
}; // OpKernel
REGISTER_KERNEL_BUILDER(Name("RoiPoolingOp").Device(DEVICE_GPU), RoiPoolingOp);




void roi_pooling_grad_gpu_launcher(int ncenters, int ngrid, int channels,
                                    const int* output_idx,
                                    const float* output_features_grad,
                                    float* input_features_grad);

class RoiPoolingGradOp: public OpKernel {
public:
    explicit RoiPoolingGradOp(OpKernelConstruction* context): OpKernel(context) {}
    void Compute(OpKernelContext* context) override {

        const Tensor& input_features = context->input(0);
        auto input_features_ptr = input_features.template flat<float>().data();
        OP_REQUIRES(context, input_features.dims()==2 && input_features.shape().dim_size(1) > 0,
                    errors::InvalidArgument("RoiPoolingGradOp expects input features in shape: [npoints, channels(>0)]."));

        const Tensor& output_idx = context->input(1);
        auto output_idx_ptr = output_idx.template flat<int>().data();
        OP_REQUIRES(context, output_idx.dims()==3 && output_idx.shape().dim_size(2) > 0,
                    errors::InvalidArgument("RoiPoolingGradOp expects output_idx in shape: [ncenters, voxel_size*3, channels(>0)]."));

        const Tensor& output_features_grad = context->input(2);
        auto output_features_grad_ptr = output_features_grad.template flat<float>().data();
        OP_REQUIRES(context, output_features_grad.dims()==3 && output_features_grad.shape().dim_size(2) > 0,
                    errors::InvalidArgument("RoiPoolingGradOp expects output_features_grad in shape: [npoints, voxel_size*3, channels(>0)]."));
        OP_REQUIRES(context, output_idx.shape().dim_size(0)==output_features_grad.shape().dim_size(0) &&
                             output_idx.shape().dim_size(1)==output_features_grad.shape().dim_size(1) &&
                             output_idx.shape().dim_size(2)==output_features_grad.shape().dim_size(2),
                             errors::InvalidArgument("RoiPoolingGradOp expects output_idx and output_features_grad has the same shape."));

        int npoint = input_features.shape().dim_size(0);
        int ncenters = output_idx.shape().dim_size(0);
        int ngrid = output_idx.shape().dim_size(1);
        int channels = output_idx.shape().dim_size(2);

        Tensor* input_features_grad = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{npoint, channels}, &input_features_grad));
        auto input_features_grad_ptr = input_features_grad->template flat<float>().data();
        cudaMemset(input_features_grad_ptr, 0.f, npoint*channels*sizeof(float));

        roi_pooling_grad_gpu_launcher(ncenters, ngrid, channels,
                                       output_idx_ptr,
                                       output_features_grad_ptr,
                                       input_features_grad_ptr);
    }
};
REGISTER_KERNEL_BUILDER(Name("RoiPoolingGradOp").Device(DEVICE_GPU), RoiPoolingGradOp);
