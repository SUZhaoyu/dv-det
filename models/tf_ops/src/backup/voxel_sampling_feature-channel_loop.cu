/* Voxel sampling GPU implementation
 * Author Zhaoyu SU
 * All Rights Reserved. Sep., 2019.
 */
#include <stdio.h>
#include <iostream>
#include <float.h>

//__global__ void output_init_gpu_kernel(int center_num, int kernel_num,
//                                       float padding, int channels,
//                                       float* output_features) {
//    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
//    if (thread_id < center_num * kernel_num) {
//        for (int c=0; c<channels; c++) {
//            output_features[thread_id*channels + c] = padding;
//        }
//    }
//}


__global__ void voxel_sampling_feature_gpu_kernel(int center_num, int channels, int kernel_num, float padding,
                                                  int output_pooling_size,
                                                  const float* input_features,
                                                  const int* output_idx,
                                                  float* output_features) {

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int voxel_coor = thread_id % kernel_num;
    if (thread_id < center_num * kernel_num) {
        int pooling_count = 0;
        for (int p=0; p<output_pooling_size; p++) {
            if (output_idx[voxel_coor*output_pooling_size + p] >= 0)
                pooling_count += 1;
        }

        for (int p=0; p<pooling_count; p++) {
            int point_id = output_idx[voxel_coor*output_pooling_size + p];
            if (point_id >= 0) {
                for (int c=0; c<channels; c++) {
                    output_features[thread_id*channels + c] += input_features[point_id * channels + c] / pooling_count;
                }
            }
        }
        if (pooling_count == 0) {
            for (int c=0; c<channels; c++) {
                output_features[thread_id*channels + c] = padding;
            }
        }
	}
}


__global__ void voxel_sampling_feature_grad_gpu_kernel(int center_num, int kernel_num, int channels,
                                                       int output_pooling_size,
                                                       const int* output_idx,
                                                       const float* output_features_grad,
                                                       float* input_features_grad) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < center_num * kernel_num) {
        int pooling_count = 0;
        for (int p=0; p<output_pooling_size; p++) {
            if (output_idx[thread_id*output_pooling_size + p] >= 0)
                pooling_count += 1;
        }

        for (int p=0; p<pooling_count; p++) {
            int point_id = output_idx[thread_id*output_pooling_size + p];
            for (int c=0; c<channels; c++) {
                atomicAdd(&input_features_grad[point_id*channels + c], output_features_grad[thread_id*channels + c] / pooling_count);
            }
        }
    }
}


void voxel_sampling_feature_gpu_launcher(int center_num, int kernel_num, int channels, float padding,
                                         int output_pooling_size,
                                         const float* input_features,
                                         const int* output_idx,
                                         float* output_features) {
    if (center_num * channels <= 0) {
        printf("VoxelSampleFeatureOp ERROR: Invalid CUDA input dimensions.\n");
        return;
    }

    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

//    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, output_init_gpu_kernel, 0, center_num * kernel_num);
//    gridSize = (center_num * kernel_num + blockSize - 1) / blockSize;
//    output_init_gpu_kernel<<<gridSize, blockSize>>>(center_num, kernel_num,
//                                                    padding, channels,
//                                                    output_features);
    float et;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, voxel_sampling_feature_gpu_kernel, 0, center_num * kernel_num);
    gridSize = (center_num * kernel_num + blockSize - 1) / blockSize;
    voxel_sampling_feature_gpu_kernel<<<gridSize, blockSize>>>(center_num, channels, kernel_num, padding,
                                                               output_pooling_size,
                                                               input_features,
                                                               output_idx,
                                                               output_features);




    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
    printf("Voxel Sampling Feature Time elapse: %f ms\n", et);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}


void voxel_sampling_feature_grad_gpu_launcher(int center_num, int kernel_num, int channels,
                                              int output_pooling_size,
                                              const int* output_idx,
                                              const float* output_features_grad,
                                              float* input_features_grad) {
    if (center_num==0 || kernel_num*channels == 0) {
        printf("VoxelSampleGradOp ERROR: Invalid CUDA input dimensions.\n");
        return;
    }
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, voxel_sampling_feature_grad_gpu_kernel, 0, center_num * kernel_num);
    gridSize = (center_num * kernel_num + blockSize - 1) / blockSize;
    voxel_sampling_feature_grad_gpu_kernel<<<gridSize, blockSize>>>(center_num, kernel_num, channels,
                                                                    output_pooling_size,
                                                                    output_idx,
                                                                    output_features_grad,
                                                                    input_features_grad);
}