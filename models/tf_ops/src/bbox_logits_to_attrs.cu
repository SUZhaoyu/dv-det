#include <stdio.h>

__global__ void bbox_logits_to_attrs_gpu_kernel(int input_npoint, int channels,
                                                const float* input_roi_attrs,
                                                const float* input_logits,
                                                float* output_attrs) {
	int input_id = threadIdx.x + blockIdx.x * blockDim.x;
	if (input_id < input_npoint) {
        float roi_diag = sqrtf(powf(input_roi_attrs[input_id * 7 + 0], 2) + powf(input_roi_attrs[input_id * 7 + 1], 2));
        output_attrs[input_id * 7 + 0] = expf(input_logits[input_id * channels + 0]) * input_roi_attrs[input_id * 7 + 0];
        output_attrs[input_id * 7 + 1] = expf(input_logits[input_id * channels + 1]) * input_roi_attrs[input_id * 7 + 1];
        output_attrs[input_id * 7 + 2] = expf(input_logits[input_id * channels + 2]) * input_roi_attrs[input_id * 7 + 2];
        output_attrs[input_id * 7 + 3] = input_logits[input_id * channels + 3] * roi_diag + input_roi_attrs[input_id * 7 + 3];
        output_attrs[input_id * 7 + 4] = input_logits[input_id * channels + 4] * roi_diag + input_roi_attrs[input_id * 7 + 4];
        output_attrs[input_id * 7 + 5] = input_logits[input_id * channels + 5] * input_roi_attrs[input_id * 7 + 2] + input_roi_attrs[input_id * 7 + 5];
        output_attrs[input_id * 7 + 6] = input_logits[input_id * channels + 6] * 3.1415927 + input_roi_attrs[input_id * 7 + 6];
    }
}


void bbox_logits_to_attrs_gpu_launcher(int input_npoint, int channels,
                                       const float* input_roi_attrs,
                                       const float* input_logits,
                                       float* output_attrs) {
    if (input_npoint == 0)
        return;
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, bbox_logits_to_attrs_gpu_kernel, 0, input_npoint);
    // Round up according to array size
    gridSize = (input_npoint + blockSize - 1) / blockSize;

    bbox_logits_to_attrs_gpu_kernel<<<gridSize, blockSize>>>(input_npoint, channels,
                                                             input_roi_attrs,
                                                             input_logits,
                                                             output_attrs);
}
