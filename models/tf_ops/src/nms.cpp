/* Voxel Sampling Operation
 * with unstructured number of input points for each mini batch
 * Created by Zhaoyu SU
 * All Rights Reserved. Sep., 2019.
 */
#include <iostream>
#include <stdio.h>
// #include <string>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <algorithm>
#include <list>
#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"


#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

using namespace tensorflow;

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;


REGISTER_OP("BoxesIou3d")
    .Input("input_boxes_a: float32")
    .Input("input_boxes_b: float32")
    .Output("output_iou_3d: float32")
    // .Attr("npoints: float")
    // .Attr("batchSize: int")
    // .Attr("nFeatures: int")
    .SetShapeFn([](InferenceContext* c){
        // ShapeHandle input_features_shape;
        // TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input_features_shape));
        // ShapeHandle roi_attrs_shape;
        // TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &roi_attrs_shape));

        // int kernel_size;
        // TF_RETURN_IF_ERROR(c->GetAttr("kernel_size", &kernel_size));

        // DimensionHandle kernel_number = c->Dim(roi_attrs_shape, 0);
        // DimensionHandle channels = c->Dim(input_features_shape, 1);

        // // The output shape during the shape inference stage is pseudo.
        // ShapeHandle output_shape = c->MakeShape({kernel_number, kernel_size * kernel_size * kernel_size, channels});

        // c->set_output(0, output_shape); // output_features
        // c->set_output(1, output_shape); // output_idx

        return Status::OK();
    }); // InferenceContext

void boxesIou3dGPUKernelLauncher(
    const int num_a,
    const float *boxes_a,
    const int num_b,
    const float *boxes_b,
    float *ans_iou);

class BoxesIou3d: public OpKernel {
public:
    explicit BoxesIou3d(OpKernelConstruction* context): OpKernel(context) {


        // OP_REQUIRES_OK(context, context->GetAttr("batchSize", &batchSize));
        // OP_REQUIRES_OK(context, context->GetAttr("nPoints", &nPoints));
        // OP_REQUIRES_OK(context, context->GetAttr("nFeatures", &nFeatures));

        // OP_REQUIRES(context, batchSize > 0,
                    // errors::InvalidArgument("Batch Size has to be larger than 0"));
        // OP_REQUIRES(context, nPoints > 0,
                    // errors::InvalidArgument("Number of points has to be larger than 0"));
        // OP_REQUIRES(context, nFeatures >= 3,
        //             errors::InvalidArgument("Number of features has to be larger than 3"));
    }
    void Compute(OpKernelContext* context) override {
        // printf("1");

        const Tensor& input_boxes_a = context->input(0);
        auto input_boxes_a_ptr = input_boxes_a.template flat<float>().data();
        OP_REQUIRES(context, input_boxes_a.dims()==2 && input_boxes_a.shape().dim_size(1)==7,
                    errors::InvalidArgument("BoxesIou3d expects boxes_a in shape: [M, 7]."));

        const Tensor& input_boxes_b = context->input(1);
        auto input_boxes_b_ptr = input_boxes_b.template flat<float>().data();
        OP_REQUIRES(context, input_boxes_b.dims()==2 && input_boxes_b.shape().dim_size(1)==7,
                    errors::InvalidArgument("BoxesIou3d expects boxes_b in shape: [N, 7]."));

        const int num_a = input_boxes_a.dim_size(0);
        const int num_b = input_boxes_b.dim_size(0);


        Tensor* output_iou3d = nullptr;
        auto output_iou3d_shape = TensorShape({num_a, num_b});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_iou3d_shape, &output_iou3d));
        float* output_iou3d_ptr = output_iou3d->template flat<float>().data();

        // printf("1");
        boxesIou3dGPUKernelLauncher(
            num_a,
            input_boxes_a_ptr,
            num_b,
            input_boxes_b_ptr,
            output_iou3d_ptr);
        // printf("1");
        // printf(std::to_string(output_bev_overlap_area_ptr[0]));
    }
private:
    // float padding_value;
    // int kernel_size, pooling_size;
}; // OpKernel
REGISTER_KERNEL_BUILDER(Name("BoxesIou3d").Device(DEVICE_GPU), BoxesIou3d);
















// params input_boxes: (N, 5) [x1, y1, x2, y2, ry]
// params output_keep_index: (N) 
// params output_num_to_keep: (1)
REGISTER_OP("RotatedNms3d")
    .Input("input_boxes: float32")
    .Output("output_keep_index: int64")
    .Output("output_num_to_keep: int32")
    .Attr("nms_overlap_thresh: float")
    // .Attr("npoints: float")
    // .Attr("batchSize: int")
    // .Attr("nFeatures: int")
    .SetShapeFn([](InferenceContext* c){
        // ShapeHandle input_features_shape;
        // TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input_features_shape));
        // ShapeHandle roi_attrs_shape;
        // TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &roi_attrs_shape));

        // int kernel_size;
        // TF_RETURN_IF_ERROR(c->GetAttr("kernel_size", &kernel_size));

        // DimensionHandle kernel_number = c->Dim(roi_attrs_shape, 0);
        // DimensionHandle channels = c->Dim(input_features_shape, 1);

        // // The output shape during the shape inference stage is pseudo.
        // ShapeHandle output_shape = c->MakeShape({kernel_number, kernel_size * kernel_size * kernel_size, channels});

        // c->set_output(0, output_shape); // output_features
        // c->set_output(1, output_shape); // output_idx

        return Status::OK();
    }); // InferenceContext

void nms3dGPUKernelLauncher(
    const int boxes_num, 
    const float nms_overlap_thresh, 
    const float *boxes, 
    unsigned long long * mask);

class RotatedNms3d: public OpKernel {
public:
    explicit RotatedNms3d(OpKernelConstruction* context): OpKernel(context) {


        OP_REQUIRES_OK(context, context->GetAttr("nms_overlap_thresh", &nms_overlap_thresh));
        // OP_REQUIRES_OK(context, context->GetAttr("nPoints", &nPoints));
        // OP_REQUIRES_OK(context, context->GetAttr("nFeatures", &nFeatures));

        // OP_REQUIRES(context, batchSize > 0,
                    // errors::InvalidArgument("Batch Size has to be larger than 0"));
        // OP_REQUIRES(context, nPoints > 0,
                    // errors::InvalidArgument("Number of points has to be larger than 0"));
        // OP_REQUIRES(context, nFeatures >= 3,
        //             errors::InvalidArgument("Number of features has to be larger than 3"));
    }
    void Compute(OpKernelContext* context) override {
        // printf("1");

        const Tensor& input_boxes = context->input(0);
        auto input_boxes_ptr = input_boxes.template flat<float>().data();
        OP_REQUIRES(context, input_boxes.dims()==2 && input_boxes.dim_size(1)==7,
                    errors::InvalidArgument("RotatedNms3d expects boxes_a in shape: [M, 7]."));

        const int num_boxes = input_boxes.dim_size(0);
        
        

        const int col_blocks = DIVUP(num_boxes, THREADS_PER_BLOCK_NMS);
        Tensor mask_buffer;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<unsigned long long>::value,
            TensorShape{num_boxes * col_blocks}, &mask_buffer));
        unsigned long long* mask_buffer_ptr = mask_buffer.template flat<unsigned long long>().data();
        cudaMemset(mask_buffer_ptr, 0, num_boxes * col_blocks*sizeof(unsigned long long));

        // printf("1");
        nms3dGPUKernelLauncher(
            num_boxes, 
            nms_overlap_thresh,
            input_boxes_ptr,
            mask_buffer_ptr);
        // printf("1");
        // printf(std::to_string(output_bev_overlap_area_ptr[0]));
        // printf("num_boxes: %d \n", num_boxes);

        // printf("boxes_num=%d, col_blocks=%d\n", num_boxes, col_blocks);
        // printf("sizeof(unsigned long long)=%lu, sizeof(uint64_t)=%lu \n", sizeof(long long int), sizeof(uint64_t));

        // 

        std::vector<unsigned long long> mask_cpu(num_boxes * col_blocks);

        cudaMemcpy(&mask_cpu[0], mask_buffer_ptr, num_boxes * col_blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        unsigned long long remv_cpu[col_blocks];
        memset(remv_cpu, 0, col_blocks * sizeof(unsigned long long));

        int output_keep_index_ptr_host_byte_size = num_boxes * sizeof(long long int);
        long long int* output_keep_index_ptr_host = (long long int*)malloc(output_keep_index_ptr_host_byte_size);
        memset(output_keep_index_ptr_host, 0, num_boxes * sizeof(long long int));

        int* num_to_keep_ptr_host = (int*)malloc(sizeof(int));
        memset(num_to_keep_ptr_host, 0, sizeof(int));

        for (int i = 0; i < num_boxes; i++){
            int nblock = i / THREADS_PER_BLOCK_NMS;
            int inblock = i % THREADS_PER_BLOCK_NMS;

            if (!(remv_cpu[nblock] & (1ULL << inblock))){
                int cur_id = num_to_keep_ptr_host[0];
                output_keep_index_ptr_host[cur_id] = (long long int)i;
                num_to_keep_ptr_host[0] += 1;
                unsigned long long *p = &mask_cpu[0] + i * col_blocks;
                for (int j = nblock; j < col_blocks; j++){
                    remv_cpu[j] |= p[j];
                }
            }
        }

        Tensor* output_keep_index = nullptr;
        auto output_keep_index_shape = TensorShape({num_boxes});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_keep_index_shape, &output_keep_index));
        long long int* output_keep_index_ptr = output_keep_index->template flat<long long int>().data();

        // printf("num to keep: %d \n", num_to_keep_ptr_host[0]);
        // printf("Successfully created output_keep_index \n");

        Tensor* output_num_to_keep = nullptr;
        auto output_num_to_keep_shape = TensorShape({1});
        OP_REQUIRES_OK(context, context->allocate_output(1, output_num_to_keep_shape, &output_num_to_keep));
        int* output_num_to_keep_ptr = output_num_to_keep->template flat<int>().data();

        cudaMemcpy(output_keep_index_ptr, output_keep_index_ptr_host, output_keep_index_ptr_host_byte_size, cudaMemcpyHostToDevice);
        cudaMemcpy(output_num_to_keep_ptr, num_to_keep_ptr_host, sizeof(int), cudaMemcpyHostToDevice);

        free(num_to_keep_ptr_host);
        free(output_keep_index_ptr_host);

    }
private:
    // float padding_value;
    // int kernel_size, pooling_size;
    float nms_overlap_thresh;
}; // OpKernel
REGISTER_KERNEL_BUILDER(Name("RotatedNms3d").Device(DEVICE_GPU), RotatedNms3d);

