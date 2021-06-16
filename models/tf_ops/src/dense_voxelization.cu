/* Voxel sampling GPU implementation
 * Author Zhaoyu SU
 * All Rights Reserved. Sep., 2019.
 */
#include <stdio.h>
#include <iostream>
#include <float.h>
#include <vector>


__device__ inline int get_batch_id(int* accu_list, int batch_size, int id) {
    for (int b=0; b<batch_size-1; b++) {
        if (id >= accu_list[b]) {
            if(id < accu_list[b+1])
                return b;
        }
    }
    return batch_size - 1;
}


__global__ void dense_voxelization_idx_gpu_kernel(int batch_size, int input_point_num,
                                                  float resolution_w, float resolution_l, float resolution_h,
                                                  int output_w, int output_l, int output_h,
                                                  const float* input_coors,
                                                  const int* input_num_list,
                                                  int* input_accu_list,
                                                  int* count_buffer,
                                                  int* output_idx) {
    const int output_voxel_size = output_w * output_l * output_h;
    int point_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (point_id < input_point_num) {
        int center_grid_coor_x = (int)floor(input_coors[point_id*3 + 0] / resolution_w);
        int center_grid_coor_y = (int)floor(input_coors[point_id*3 + 1] / resolution_l);
        int center_grid_coor_z = (int)floor(input_coors[point_id*3 + 2] / resolution_h);
        int batch_id = get_batch_id(input_accu_list, batch_size, point_id);
        int voxel_idx = batch_id * output_voxel_size + center_grid_coor_x * output_l * output_h + center_grid_coor_y * output_h + center_grid_coor_z;
        int count = atomicAdd(&count_buffer[voxel_idx], 1);
        if (count < 1) {
            output_idx[voxel_idx] = point_id;
        }
    }
}


__global__ void dense_voxelization_features_gpu_kernel(int batch_size, int channels,
                                                       int output_w, int output_l, int output_h,
                                                       const float* input_features,
                                                       float* output_features,
                                                       int* count_buffer,
                                                       int* output_idx) {
    int voxel_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (voxel_id < batch_size * output_w * output_l * output_h) {
        int count = count_buffer[voxel_id];
        if (count > 0) {
            int point_id = output_idx[voxel_id];\
            for (int c = 0; c < channels; c++) {
                output_features[voxel_id * channels + c] = input_features[point_id * channels + c];
//                output_features[voxel_id * channels + c] = 1.;
            }
        }
    }
}


__global__ void dense_voxelization_grad_gpu_kernel(int batch_size, int channels,
                                                   int output_w, int output_l, int output_h,
                                                   const float* output_features_grad,
                                                   const int* output_idx,
                                                   float* input_features_grad) {
    int voxel_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (voxel_id < batch_size * output_w * output_l * output_h) {
        int point_id = output_idx[voxel_id];
        if (point_id >= 0) {
            for (int c = 0; c < channels; c++) {
                input_features_grad[point_id * channels + c] = output_features_grad[voxel_id + c];
            }
        }
    }
}


void dense_voxelization_gpu_launcher(int batch_size, int input_point_num, int channels,
                                     std::vector<float> resolution, std::vector<int> output_size,
                                     const float* input_coors,
                                     const float* input_features,
                                     const int* input_num_list,
                                     int* input_accu_list,
                                     int* count_buffer,
                                     float* output_features,
                                     int* output_idx) {
    if (batch_size*input_point_num <=0) {
        printf("BevProjectionOp ERROR: Invalid CUDA input dimensions: [%d, %d]\n", batch_size, input_point_num);
        return;
    }

    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size


    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dense_voxelization_idx_gpu_kernel, 0, input_point_num);
    gridSize = (input_point_num + blockSize - 1) / blockSize;
    dense_voxelization_idx_gpu_kernel<<<gridSize, blockSize>>>(batch_size, input_point_num,
                                                               resolution[0], resolution[1], resolution[2],
                                                               output_size[0], output_size[1], output_size[2],
                                                               input_coors,
                                                               input_num_list,
                                                               input_accu_list,
                                                               count_buffer,
                                                               output_idx);


    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dense_voxelization_features_gpu_kernel, 0, batch_size * output_size[0] * output_size[1] * output_size[2]);
    gridSize = (batch_size * output_size[0] * output_size[1] * output_size[2] + blockSize - 1) / blockSize;
    dense_voxelization_features_gpu_kernel<<<gridSize, blockSize>>>(batch_size, channels,
                                                                    output_size[0], output_size[1], output_size[2],
                                                                    input_features,
                                                                    output_features,
                                                                    count_buffer,
                                                                    output_idx);
}

void dense_voxelization_grad_gpu_launcher(int batch_size, int channels, std::vector<int> output_size,
                                          const float* output_features_grad,
                                          const int* output_idx,
                                          float* input_features_grad) {
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dense_voxelization_grad_gpu_kernel, 0, batch_size * output_size[0] * output_size[1] * output_size[2]);
    gridSize = (batch_size * output_size[0] * output_size[1] * output_size[2] + blockSize - 1) / blockSize;
    dense_voxelization_grad_gpu_kernel<<<gridSize, blockSize>>>(batch_size, channels,
                                                                output_size[0], output_size[1], output_size[2],
                                                                output_features_grad,
                                                                output_idx,
                                                                input_features_grad);

}

