#include "tensorflow/core/platform/stream_executor.h"

using namespace tensorflow;


template <typename T>
se::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory, uint64 size) {
    se::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory), size * sizeof(T));
    se::DeviceMemory<T> typed(wrapped);
    return typed;
}

template <typename T>
struct GemmLauncher {
	static void launch(OpKernelContext* ctx,
					   const Tensor& A,
					   const Tensor& B,
					   Tensor* C,
					   const uint64 m,
					   const uint64 k,
					   const uint64 n,
					   bool trans_A,
					   bool trans_B) {
		auto* stream = ctx->op_device_context()->stream();
		OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

		auto a_ptr = AsDeviceMemory(A.template flat<T>().data(), A.template flat<T>().size());
		auto b_ptr = AsDeviceMemory(B.template flat<T>().data(), B.template flat<T>().size());
		auto c_ptr = AsDeviceMemory(C->template flat<T>().data(), C->template flat<T>().size());

		auto adj_a = trans_A ? se::blas::Transpose::kTranspose : se::blas::Transpose::kNoTranspose;
		auto adj_b = trans_B ? se::blas::Transpose::kTranspose : se::blas::Transpose::kNoTranspose;

		bool blas_launch_status = stream->ThenBlasGemm(adj_b, 
													   adj_a, 
													   n, 
													   m, 
													   k, 
													   1.0f,
													   b_ptr, 
													   trans_B ? k : n,
													   a_ptr, 
													   trans_A ? m : k,
													   0.0f, 
													   &c_ptr, 
													   n).ok();
		// execute order for Gemm: a->[m x k], b->[k x n]; output: c->[m x n]
		if (!blas_launch_status) {
			ctx->SetStatus(errors::Internal("Blas SGEMM launch failed : A.shape=", A.shape().DebugString(),
																	 " ,B.shape=", B.shape().DebugString(),
																	 " ,C.shape=", C->shape().DebugString()));
		}
	} // void launch
}; // struc FeatureFuseLauncher