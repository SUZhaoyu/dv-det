/* Voxel sampling GPU implementation
 * Author Zhaoyu SU
 * All Rights Reserved. Sep., 2019.
 */
#include <stdio.h>
#include <iostream>
#include <float.h>  // import FLT_EPSILON

__global__ void voxel2col_gpu_kernel(int input_num, int channels, int input_voxel_size,
                                      int output_voxel_size, int kernel_size,
                                      const float* input_voxels,
                                      float* output_voxels,
                                      int* output_idx) {

    const int input_voxel_num = input_voxel_size * input_voxel_size * input_voxel_size;
    const int output_voxel_num = output_voxel_size * output_voxel_size * output_voxel_size;
    const int kernel_num = kernel_size * kernel_size * kernel_size;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < input_num * output_voxel_num * kernel_num) {
        output_idx[thread_id] = -1;
    }
    __syncthreads();
    if (thread_id < input_num * output_voxel_num * kernel_num) {
        int current_id = thread_id / (output_voxel_num * kernel_num);
        int output_voxel_id = thread_id % (output_voxel_num * kernel_num) / kernel_num;
        int kernel_id = thread_id % kernel_num;

        int kernel_x = kernel_id / (kernel_size * kernel_size);
        int kernel_y = kernel_id % (kernel_size * kernel_size) / kernel_size;
        int kernel_z = kernel_id % kernel_size;

        int output_voxel_coor_x = output_voxel_id / (output_voxel_size * output_voxel_size);
        int output_voxel_coor_y = output_voxel_id % (output_voxel_size * output_voxel_size) / output_voxel_size;
        int output_voxel_coor_z = output_voxel_id % output_voxel_size;

        /*
        input_voxel_coor = output_voxel_coor + 1;
        kernel_coor = input_voxel_coor + [kernel_x/y/z] - 1, for kernel_x/y/z in [0, 1, 2];
        so:
        kernel_coor = output_voxel_coor + [kernel_x/y/z], for kernel_x/y/z in [0, 1, 2];
        */

        int kernel_coor_x = output_voxel_coor_x + kernel_x;
        int kernel_coor_y = output_voxel_coor_y + kernel_y;
        int kernel_coor_z = output_voxel_coor_z + kernel_z;
        int input_voxel_id = current_id * input_voxel_num + \
                             kernel_coor_x * input_voxel_size * input_voxel_size + \
                             kernel_coor_y * input_voxel_size + \
                             kernel_coor_z;
        output_idx[thread_id] = input_voxel_id;
        for (int c=0; c<channels; c++) {
            output_voxels[thread_id * channels + c] = input_voxels[input_voxel_id * channels + c];
        }
    }
}


__global__ void voxel2col_grad_gpu_kernel(int input_num, int output_voxel_num, int kernel_num, int channels,
                                          const float* input_voxels,
                                          const int* output_idx,
                                          const float* output_voxels_grad,
                                          float* input_voxels_grad) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < input_num * output_voxel_num * kernel_num) {
        int input_voxel_id = output_idx[thread_id];
        if (input_voxel_id >= 0) {
            for (int c=0; c<channels; c++) {
                atomicAdd(&input_voxels_grad[input_voxel_id*channels + c], output_voxels_grad[thread_id*channels + c]);
            }
        }
    }
}



void voxel2col_gpu_launcher(int input_num, int channels, int input_voxel_size,
                             int output_voxel_size, int kernel_size,
                             const float* input_voxels,
                             float* output_voxels,
                             int* output_idx) {
    if (channels * input_voxel_size * output_voxel_size * kernel_size <= 0) {
        printf("Voxel2ColOp ERROR: Invalid CUDA input dimensions.\n");
        return;
    }
    if (input_num <=0) {
        return;
    }
    const int output_voxel_num = output_voxel_size * output_voxel_size * output_voxel_size;
    const int kernel_num = kernel_size * kernel_size * kernel_size;
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, voxel2col_gpu_kernel, 0, input_num * output_voxel_num * kernel_num);
    gridSize = (input_num * output_voxel_num * kernel_num + blockSize - 1) / blockSize;

    voxel2col_gpu_kernel<<<gridSize, blockSize>>>(input_num, channels, input_voxel_size,
                                                  output_voxel_size, kernel_size,
                                                  input_voxels,
                                                  output_voxels,
                                                  output_idx);
}

void voxel2col_grad_gpu_launcher(int input_num, int output_voxel_num, int kernel_num, int channels,
                                 const float* input_voxels,
                                 const int* output_idx,
                                 const float* output_voxels_grad,
                                 float* input_voxels_grad) {
    if (channels * output_voxel_num * kernel_num <= 0) {
        printf("Voxel2ColGradOp ERROR: Invalid CUDA input dimensions.\n");
        return;
    }
    if (input_num <=0) {
        return;
    }

    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, voxel2col_gpu_kernel, 0, input_num * output_voxel_num * kernel_num);
    gridSize = (input_num * output_voxel_num * kernel_num + blockSize - 1) / blockSize;

    voxel2col_grad_gpu_kernel<<<gridSize, blockSize>>>(input_num, output_voxel_num, kernel_num, channels,
                                                       input_voxels,
                                                       output_idx,
                                                       output_voxels_grad,
                                                       input_voxels_grad);


}