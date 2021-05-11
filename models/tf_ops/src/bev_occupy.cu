/* Voxel sampling GPU implementation
 * Author Zhaoyu SU
 * All Rights Reserved. Sep., 2019.
 */
#include <stdio.h>
#include <iostream>
#include <float.h>


__device__ int get_batch_id(int* accu_list, int batch_size, int id) {
    for (int b=0; b<batch_size-1; b++) {
        if (id >= accu_list[b]) {
            if(id < accu_list[b+1])
                return b;
        }
    }
    return batch_size - 1;
}


__global__ void bev_occupy_gpu_kernel(int batch_size, int input_point_num,
                                      int output_w, int output_l, float resolution,
                                      const float* input_coors,
                                      const int* input_num_list,
                                      int* input_accu_list,
                                      int* output_occupy) {
    const int output_img_size = output_w * output_l;
    int point_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (point_id < input_point_num) {
        int center_grid_coor_x = (int)floor(input_coors[point_id*3 + 0] / resolution);
        int center_grid_coor_y = (int)floor(input_coors[point_id*3 + 1] / resolution);
        int batch_id = get_batch_id(input_accu_list, batch_size, point_id);
        int output_idx = batch_id * output_img_size + center_grid_coor_x * output_l + center_grid_coor_y;
        atomicAdd(&output_occupy[output_idx], 1);
    }
}



void bev_occupy_gpu_launcher(int batch_size, int input_point_num,
                             int output_w, int output_l, float resolution,
                             const float* input_coors,
                             const int* input_num_list,
                             int* input_accu_list,
                             int* output_occupy) {
    if (batch_size*input_point_num <=0) {
        printf("BevOccupyOp ERROR: Invalid CUDA input dimensions.\n");
        return;
    }

    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size


    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, bev_occupy_gpu_kernel, 0, input_point_num);
    gridSize = (input_point_num + blockSize - 1) / blockSize;
    bev_occupy_gpu_kernel<<<gridSize, blockSize>>>(batch_size, input_point_num,
                                                   output_w, output_l, resolution,
                                                   input_coors,
                                                   input_num_list,
                                                   input_accu_list,
                                                   output_occupy);
}

