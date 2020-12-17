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
//    const float PI = 3.1415927;
    const int input_voxel_num = input_voxel_size * input_voxel_size * input_voxel_size;
    const int output_voxel_num = output_voxel_size * output_voxel_size * output_voxel_size;
    const int kernel_num = kernel_size * kernel_size * kernel_size;
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < input_num * output_voxel_num * kernel_num) {
        output_idx[id] = -1;
    }
    __syncthreads();
    if (id < input_num * output_voxel_num * kernel_num) {
        int input_id = id / (output_voxel_num * kernel_num);
        int voxel_id = id % (input_id * kernel_num) / kernel_num;
        int kernel_id = id % kernel_num;

        int z = kernel_id / (kernel_size * kernel_size);
        int y = kernel_id % (kernel_size * kernel_size) / kernel_size;
        int x = kernel_id % kernel_size;

        int output_voxel_coor_z = voxel_id / (output_voxel_size * output_voxel_size);
        int output_voxel_coor_y = voxel_id % (output_voxel_size * output_voxel_size) / output_voxel_size;
        int output_voxel_coor_x = voxel_id % output_voxel_size;

        /*
        input_voxel_coor = output_voxel_coor + 1;
        kernel_coor = input_voxel_coor + [x/y/z] - 1, for x/y/z in [0, 1, 2];
        so:
        kernel_coor = output_voxel_coor + [x/y/z], for x/y/z in [0, 1, 2];
        */

        int kernel_coor_z = output_voxel_coor_z + z;
        int kernel_coor_y = output_voxel_coor_y + y;
        int kernel_coor_x = output_voxel_coor_x + x;
        int input_voxel_id = input_id * input_voxel_num + \
                             kernel_coor_z * input_voxel_size * input_voxel_size + \
                             kernel_coor_y * input_voxel_size + \
                             kernel_coor_x;
        int output_kernel_id = input_id * output_voxel_num * kernel_num + voxel_id * kernel_num + z * kernel_size * kernel_size + y * kernel_size + x;
        output_idx[output_kernel_id] = input_voxel_id;
        for (int c=0; c<channels; c++) {
            output_voxels[output_kernel_id * channels + c] = input_voxels[input_voxel_id * channels + c];
        }

    }
}



void voxel2col_gpu_launcher(int input_num, int channels, int input_voxel_size,
                             int output_voxel_size, int kernel_size,
                             const float* input_voxels,
                             float* output_voxels,
                             int* output_idx) {
    if (input_num*channels <=0 || input_voxel_size * output_voxel_size * kernel_size <= 0) {
        printf("DenseConvOp ERROR: Invalid CUDA input dimensions.\n");
        return;
    }
    const int output_voxel_num = output_voxel_size * output_voxel_size * output_voxel_size;
    const int kernel_num = kernel_size * kernel_size * kernel_size;
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, voxel2col_gpu_kernel, 0, input_num * output_voxel_num * kernel_num);
    gridSize = (input_num * output_voxel_num * kernel_num + blockSize - 1) / blockSize;

    voxel2col_gpu_kernel<<<gridSize,blockSize>>>(input_num, channels, input_voxel_size,
                                                  output_voxel_size, kernel_size,
                                                  input_voxels,
                                                  output_voxels,
                                                  output_idx);
}
