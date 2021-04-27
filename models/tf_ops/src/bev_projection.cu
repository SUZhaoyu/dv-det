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


__global__ void bev_projection_idx_gpu_kernel(int batch_size, int input_point_num,
                                              int output_w, int output_l, float resolution, int buffer_size,
                                              const float* input_coors,
                                              const int* input_num_list,
                                              int* input_accu_list,
                                              int* count_buffer,
                                              int* idx_buffer,
                                              int* output_idx) {
    const int output_img_size = output_w * output_l;
    int point_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (point_id < input_point_num) {
        int center_grid_coor_x = (int)floor(input_coors[point_id*3 + 0] / resolution);
        int center_grid_coor_y = (int)floor(input_coors[point_id*3 + 1] / resolution);
        int batch_id = get_batch_id(input_accu_list, batch_size, point_id);
        int img_buffer_idx = batch_id * output_img_size + center_grid_coor_x * output_l + center_grid_coor_y;
        int count = atomicAdd(&count_buffer[img_buffer_idx], 1);
        if (count < buffer_size) {
            idx_buffer[img_buffer_idx*buffer_size + count] = point_id;
        }
    }
}


__global__ void bev_projection_features_gpu_kernel(int batch_size, int output_w, int output_l, int channels, int buffer_size,
                                                   const float* input_features,
                                                   float* output_features,
                                                   int* count_buffer,
                                                   int* idx_buffer,
                                                   int* output_idx) {
    int pixel_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (pixel_id < batch_size * output_w * output_l) {
        int count = count_buffer[pixel_id];
        if (count > 0) {
            int buffer_start_id = pixel_id * buffer_size;
            for (int c = 0; c < channels; c++) {
                int pixel_channel_id = pixel_id * channels + c;
                float max_f = -1e7;
                for (int i = 0; i < count; i++) {
                    int point_id = idx_buffer[buffer_start_id + i];
                    if (point_id >= 0) {
                        float f = input_features[point_id * channels + c];
                        if (f > max_f) {
                            max_f = f;
                            output_features[pixel_channel_id] = f;
                            output_idx[pixel_channel_id] = point_id;
                        }
                    }
                }
            }
        }
    }
}


__global__ void bev_projection_grad_gpu_kernel(int batch_size, int output_w, int output_l, int channels,
                                               const float* output_features_grad,
                                               const int* output_idx,
                                               float* input_features_grad) {
    int pixel_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (pixel_id < batch_size * output_w * output_l) {
        int pixel_start_id = pixel_id * channels;
        for (int c = 0; c < channels; c++) {
            int point_id = output_idx[pixel_start_id + c];
            if (point_id >= 0) {
                input_features_grad[point_id * channels + c] = output_features_grad[pixel_start_id + c];
            }
        }
    }
}


void bev_projection_gpu_launcher(int batch_size, int input_point_num, int channels,
                                 int output_w, int output_l, float resolution, int buffer_size,
                                 const float* input_coors,
                                 const float* input_features,
                                 const int* input_num_list,
                                 int* input_accu_list,
                                 int* count_buffer,
                                 int* idx_buffer,
                                 float* output_features,
                                 int* output_idx) {
    if (batch_size*input_point_num <=0) {
        printf("BevProjectionOp ERROR: Invalid CUDA input dimensions.\n");
        return;
    }

    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size


    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, bev_projection_idx_gpu_kernel, 0, input_point_num);
    gridSize = (input_point_num + blockSize - 1) / blockSize;
    bev_projection_idx_gpu_kernel<<<gridSize, blockSize>>>(batch_size, input_point_num,
                                                           output_w, output_l, resolution, buffer_size,
                                                           input_coors,
                                                           input_num_list,
                                                           input_accu_list,
                                                           count_buffer,
                                                           idx_buffer,
                                                           output_idx);


    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, bev_projection_features_gpu_kernel, 0, batch_size * output_w * output_l);
    gridSize = (batch_size * output_w * output_l + blockSize - 1) / blockSize;
    bev_projection_features_gpu_kernel<<<gridSize, blockSize>>>(batch_size, output_w, output_l, channels, buffer_size,
                                                                input_features,
                                                                output_features,
                                                                count_buffer,
                                                                idx_buffer,
                                                                output_idx);
}

void bev_projection_grad_gpu_launcher(int batch_size, int output_w, int output_l, int channels,
                                      const float* output_features_grad,
                                      const int* output_idx,
                                      float* input_features_grad) {
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, bev_projection_grad_gpu_kernel, 0, batch_size * output_w * output_l);
    gridSize = (batch_size * output_w * output_l + blockSize - 1) / blockSize;
    bev_projection_grad_gpu_kernel<<<gridSize, blockSize>>>(batch_size, output_w, output_l, channels,
                                                            output_features_grad,
                                                            output_idx,
                                                            input_features_grad);

}

