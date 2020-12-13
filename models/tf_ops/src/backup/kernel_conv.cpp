#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"

#include "gemm_utils.h"

using namespace tensorflow;

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("KernelConvOp")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {half, float, double}")
    .SetShapeFn([](InferenceContext * c) {
        ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input_shape)); // [nkernel, ngrid, input_channels]
        ShapeHandle filter_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &filter_shape)); // [ngrid, input_channels, output_channels]

        DimensionHandle nkernel = c->Dim(input_shape, 0);
        DimensionHandle ngrid = c->Dim(input_shape, 1);
        DimensionHandle input_channels = c->Dim(input_shape, 2);
        DimensionHandle output_channels = c->Dim(filter_shape, 2);

        ShapeHandle output_shape = c->MakeShape({nkernel, output_channels});
        c->set_output(0, output_shape);
        return Status::OK();
    });

REGISTER_OP("KernelConvGradOp")
	.Input("input: T")
	.Input("filter: T")
	.Input("output_grad: T")
	.Output("input_grad: T")
	.Output("filter_grad: T")
	.Attr("T: {half, float, double}")
	.SetShapeFn([](InferenceContext * c) {
		c->set_output(0, c->input(0));
		c->set_output(1, c->input(1));
		return Status::OK();
	});


template <typename Device, typename T>
class KernelConvOp: public OpKernel {
public:
	explicit KernelConvOp(OpKernelConstruction * context) : OpKernel(context) {}
	void Compute(OpKernelContext* context) override {
		const Tensor& input = context->input(0);
		const Tensor& filter = context->input(1);
		OP_REQUIRES(context, input.dims()==3,
			errors::InvalidArgument("Voxel conv expects input shape: [kernel_number, grid_number, input_channel]"));
		OP_REQUIRES(context, filter.dims()==3,
			errors::InvalidArgument("Voxel conv expects filter shape: [grid_number, input_channel, output_channel]"));
		OP_REQUIRES(context, input.dim_size(1) == filter.dim_size(0),
			errors::InvalidArgument("Grid number of input and filter does not match."));
		OP_REQUIRES(context, input.dim_size(2) == filter.dim_size(1),
			errors::InvalidArgument("Channel of input and filter does not match."));

		int nkernel = input.dim_size(0);
		int ngrid = input.dim_size(1);
		int input_channels = input.dim_size(2);
		int output_channels = filter.dim_size(2);

		Tensor* output = nullptr;
		auto output_shape = TensorShape({nkernel, output_channels});
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
		if (output_shape.num_elements() == 0) return;

		GemmLauncher<T>::launch(context, input, filter, output, nkernel, ngrid*input_channels, output_channels, false, false);
	} // Compute override
}; // class VoxelConv

REGISTER_KERNEL_BUILDER(
    Name("KernelConvOp").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    KernelConvOp<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(
    Name("KernelConvOp").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    KernelConvOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("KernelConvOp").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    KernelConvOp<GPUDevice, double>);


template <typename Device, typename T>
class KernelConvGradOp: public OpKernel {
public:
	explicit KernelConvGradOp(OpKernelConstruction * context) : OpKernel(context) {}
	void Compute(OpKernelContext* context) override {
		const Tensor& input = context->input(0);
		const Tensor& filter = context->input(1);
		const Tensor& output_grad = context->input(2);
		OP_REQUIRES(context, input.dims()==3,
			errors::InvalidArgument("Voxel conv expects input shape: [kernel_number, grid_number, channel]"));
		OP_REQUIRES(context, filter.dims()==3,
			errors::InvalidArgument("Voxel conv expects filter shape: [grid_number, input_channel, output_channel]"));
		OP_REQUIRES(context, input.dim_size(1) == filter.dim_size(0),
			errors::InvalidArgument("Grid number of input and filter does not match."));
		OP_REQUIRES(context, input.dim_size(2) == filter.dim_size(1),
			errors::InvalidArgument("Channel of input and filter does not match."));
		OP_REQUIRES(context, output_grad.dim_size(1) == filter.dim_size(2),
			errors::InvalidArgument("Channel of output_grad and filter does not match."));

		int nkernel = input.dim_size(0);
		int ngrid = input.dim_size(1);
		int input_channels = input.dim_size(2);
		int output_channels = filter.dim_size(2);

		Tensor* input_grad = nullptr;
		TensorShape input_grad_shape = input.shape();
		OP_REQUIRES_OK(context, context->allocate_output(0, input_grad_shape, &input_grad));

		Tensor* filter_grad = nullptr;
		TensorShape filter_grad_shape = filter.shape();
		OP_REQUIRES_OK(context, context->allocate_output(1, filter_grad_shape, &filter_grad));

		GemmLauncher<T>::launch(context, output_grad, filter, input_grad, nkernel, output_channels, ngrid*input_channels, false, true);
		GemmLauncher<T>::launch(context, input, output_grad, filter_grad, ngrid*input_channels, nkernel, output_channels, true, false);
	} // Compute override
}; // class VoxelConv

REGISTER_KERNEL_BUILDER(
    Name("KernelConvGradOp").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    KernelConvGradOp<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(
    Name("KernelConvGradOp").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    KernelConvGradOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("KernelConvGradOp").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    KernelConvGradOp<GPUDevice, double>);

