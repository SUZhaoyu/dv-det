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

__global__ void output_init_gpu_kernel(int batch_size, int center_num, int kernel_num,
                                       int* output_idx) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < center_num * kernel_num) {
        output_idx[thread_id] = -1;
    }
}

__global__ void grid_buffer_init_gpu_kernel(int batch_size, int input_point_num, float resolution,
                                            int grid_dim_w, int grid_dim_l, int grid_dim_h,
                                            const float* input_coors,
                                            int* input_accu_list,
                                            int* grid_buffer) {
    const int grid_dim_size = grid_dim_w * grid_dim_h * grid_dim_l;
    int point_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (point_id < input_point_num) {
        int center_grid_coor_x = (int)floor(input_coors[point_id*3 + 0] / resolution);
        int center_grid_coor_y = (int)floor(input_coors[point_id*3 + 1] / resolution);
        int center_grid_coor_z = (int)floor(input_coors[point_id*3 + 2] / resolution);
        int batch_id = get_batch_id(input_accu_list, batch_size, point_id);
        int grid_buffer_idx = batch_id * grid_dim_size + center_grid_coor_x * grid_dim_l * grid_dim_h + center_grid_coor_y * grid_dim_h + center_grid_coor_z;
        atomicExch(&grid_buffer[grid_buffer_idx], point_id);
    }
}


__global__ void voxel_sampling_idx_gpu_kernel(int batch_size, int center_num,
                                              int kernel_size,
                                              int grid_dim_w, int grid_dim_l, int grid_dim_h,
                                              float resolution,
                                              const float* input_coors,
                                              const float* center_coors,
                                              int* center_accu_list,
                                              int* grid_buffer,
                                              int* output_idx) {

	const int kernel_num = kernel_size * kernel_size * kernel_size;
	const int half_kernel_size = (kernel_size - 1) / 2;
	const int half_kernel_num = kernel_size * kernel_size * half_kernel_size + \
                                kernel_size * half_kernel_size + \
                                half_kernel_size;
	const int search_kernel_size = kernel_size + 1;
	const int search_kernel_num = search_kernel_size * search_kernel_size * search_kernel_size;
    const int grid_dim_size = grid_dim_w * grid_dim_l * grid_dim_h;
	const float radius = 1.5 * resolution;
	const float r2 = radius * radius;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < center_num * search_kernel_num) {

        int center_id = thread_id / search_kernel_num;
        int search_grid_id = thread_id % search_kernel_num;
        int batch_id = get_batch_id(center_accu_list, batch_size, center_id);

        float center_coor_x = center_coors[center_id*3 + 0];
        float center_coor_y = center_coors[center_id*3 + 1];
        float center_coor_z = center_coors[center_id*3 + 2];
        int center_grid_coor_x = __float2int_rz(center_coor_x / resolution);
        int center_grid_coor_y = __float2int_rz(center_coor_y / resolution);
        int center_grid_coor_z = __float2int_rz(center_coor_z / resolution);


        int search_grid_x = search_grid_id / (search_kernel_size * search_kernel_size);
        int search_grid_y = search_grid_id % (search_kernel_size * search_kernel_size) / search_kernel_size;
        int search_grid_z = search_grid_id % search_kernel_size;

        int search_offset_x = -2 + round(center_coor_x / resolution - center_grid_coor_x) + search_grid_x;
        int search_offset_y = -2 + round(center_coor_y / resolution - center_grid_coor_y) + search_grid_y;
        int search_offset_z = -2 + round(center_coor_z / resolution - center_grid_coor_z) + search_grid_z;

        int target_grid_x = max(0, min(center_grid_coor_x + search_offset_x, grid_dim_w - 1));
        int target_grid_y = max(0, min(center_grid_coor_y + search_offset_y, grid_dim_l - 1));
        int target_grid_z = max(0, min(center_grid_coor_z + search_offset_z, grid_dim_h - 1));
        int target_grid_id = batch_id * grid_dim_size + target_grid_x * grid_dim_l * grid_dim_h + target_grid_y * grid_dim_h + target_grid_z;
        int point_id = grid_buffer[target_grid_id];

        if (point_id>=0) {
            float coor_x = input_coors[point_id*3 +0];
            float coor_y = input_coors[point_id*3 +1];
            float coor_z = input_coors[point_id*3 +2];
            float dx = coor_x - center_coor_x + FLT_EPSILON;
            float dy = coor_y - center_coor_y + FLT_EPSILON;
            float dz = coor_z - center_coor_z + FLT_EPSILON;
            float dx2 = dx * dx;
            float dy2 = dy * dy;
            float dz2 = dz * dz;
            if (dx2 < r2 && dy2 < r2 && dz2 < r2) {
                int kernel_coor_x = __float2int_rz(dx / resolution + 0.5 * fabsf(dx) / dx);
                int kernel_coor_y = __float2int_rz(dy / resolution + 0.5 * fabsf(dy) / dy);
                int kernel_coor_z = __float2int_rz(dz / resolution + 0.5 * fabsf(dz) / dz);
                int voxel_coor = center_id * kernel_num + half_kernel_num + \
                                 kernel_size * kernel_size * kernel_coor_x + \
                                 kernel_size * kernel_coor_y + \
                                 kernel_coor_z;
                if (output_idx[voxel_coor] < 0) {
                    output_idx[voxel_coor] = point_id;
                }
            }
        }
	}
}


void voxel_sampling_idx_gpu_launcher(int batch_size, int input_point_num,
                                     int center_num, int kernel_size,
                                     int grid_dim_w, int grid_dim_l, int grid_dim_h,
                                     float resolution,
                                     const float* input_coors,
                                     const int* input_num_list,
                                     const float* center_coors,
                                     const int* center_num_list,
                                     int* input_accu_list,
                                     int* center_accu_list,
                                     int* grid_buffer,
                                     int* output_idx) {
    if (batch_size*input_point_num <=0 || center_num <= 0) {
        printf("VoxelSampleOp ERROR: Invalid CUDA input dimensions.\n");
        return;
    }
    int kernel_num = kernel_size * kernel_size * kernel_size;
    int search_kernel_num = (kernel_size + 1) * (kernel_size + 1) * (kernel_size + 1);

    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, output_init_gpu_kernel, 0, center_num * kernel_num);
    gridSize = (center_num * kernel_num + blockSize - 1) / blockSize;
    output_init_gpu_kernel<<<gridSize, blockSize>>>(batch_size, center_num, kernel_num,
                                                    output_idx);

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, grid_buffer_init_gpu_kernel, 0, input_point_num);
    gridSize = (input_point_num + blockSize - 1) / blockSize;
    grid_buffer_init_gpu_kernel<<<gridSize, blockSize>>>(batch_size, input_point_num, resolution,
                                                         grid_dim_w, grid_dim_l, grid_dim_h,
                                                         input_coors,
                                                         input_accu_list,
                                                         grid_buffer);

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, voxel_sampling_idx_gpu_kernel, 0, center_num * search_kernel_num);
    gridSize = (center_num * search_kernel_num + blockSize - 1) / blockSize;
    voxel_sampling_idx_gpu_kernel<<<gridSize, blockSize>>>(batch_size, center_num,
                                                           kernel_size,
                                                           grid_dim_w, grid_dim_l, grid_dim_h, resolution,
                                                           input_coors,
                                                           center_coors,
                                                           center_accu_list,
                                                           grid_buffer,
                                                           output_idx);
}
