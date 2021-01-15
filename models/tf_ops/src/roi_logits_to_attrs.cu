#include <stdio.h>

__global__ void roi_logits_to_attrs_gpu_kernel(int input_npoint, int channels,
                                               float anchor_w, float anchor_l, float anchor_h,
                                               const float* base_coors,
                                               const float* input_logits,
                                               float* output_attrs) {
    const float anchor_diag = sqrtf(powf(anchor_w, 2) + powf(anchor_l, 2));
	int input_id = threadIdx.x + blockIdx.x * blockDim.x;
	if (input_id < input_npoint) {
        output_attrs[input_id * 7 + 0] = expf(input_logits[input_id * channels + 0]) * anchor_w;
        output_attrs[input_id * 7 + 1] = expf(input_logits[input_id * channels + 1]) * anchor_l;
        output_attrs[input_id * 7 + 2] = expf(input_logits[input_id * channels + 2]) * anchor_h;
        output_attrs[input_id * 7 + 3] = input_logits[input_id * channels + 3] * anchor_diag + base_coors[input_id * 3 + 0];
        output_attrs[input_id * 7 + 4] = input_logits[input_id * channels + 4] * anchor_diag + base_coors[input_id * 3 + 1];
        output_attrs[input_id * 7 + 5] = input_logits[input_id * channels + 5] * anchor_h + base_coors[input_id * 3 + 2];
        output_attrs[input_id * 7 + 6] = input_logits[input_id * channels + 6] * 3.1415927;
    }
}


void roi_logits_to_attrs_gpu_launcher(int input_npoint, int channels,
                                      float anchor_w, float anchor_l, float anchor_h,
                                      const float* base_coors,
                                      const float* input_logits,
                                      float* output_attrs) {
    if (input_npoint == 0)
        return;
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, roi_logits_to_attrs_gpu_kernel, 0, input_npoint);
    // Round up according to array size
    gridSize = (input_npoint + blockSize - 1) / blockSize;

    roi_logits_to_attrs_gpu_kernel<<<gridSize, blockSize>>>(input_npoint, channels,
                                                            anchor_w, anchor_l, anchor_h,
                                                            base_coors,
                                                            input_logits,
                                                            output_attrs);
}
