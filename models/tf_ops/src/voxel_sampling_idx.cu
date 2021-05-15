/* Voxel sampling GPU implementation
 * Author Zhaoyu SU
 * All Rights Reserved. Sep., 2019.
 */
#include <stdio.h>
#include <iostream>
#include <vector>
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
                                       int output_pooling_size,
                                       int* output_idx) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < center_num * kernel_num) {
        for (int p=0; p<output_pooling_size; p++) {
            output_idx[thread_id*output_pooling_size + p] = -1;
        }
    }
}

__global__ void grid_buffer_init_gpu_kernel(int batch_size, int input_point_num,
                                            int grid_dim_w, int grid_dim_l, int grid_dim_h,
                                            float resolution_w, float resolution_l, float resolution_h,
                                            int grid_buffer_size,
                                            const float* input_coors,
                                            int* input_accu_list,
                                            int* grid_buffer,
                                            int* grid_buffer_count) {
    const int grid_dim_size = grid_dim_w * grid_dim_h * grid_dim_l;
    int point_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (point_id < input_point_num) {
        int center_grid_coor_x = (int)floor(input_coors[point_id*3 + 0] / resolution_w);
        int center_grid_coor_y = (int)floor(input_coors[point_id*3 + 1] / resolution_l);
        int center_grid_coor_z = (int)floor(input_coors[point_id*3 + 2] / resolution_h);
        int batch_id = get_batch_id(input_accu_list, batch_size, point_id);
        int grid_buffer_idx = batch_id * grid_dim_size + center_grid_coor_x * grid_dim_l * grid_dim_h + center_grid_coor_y * grid_dim_h + center_grid_coor_z;
        int count = atomicAdd(&grid_buffer_count[grid_buffer_idx], 1);
//        printf("%d\n", count);
        if (count < grid_buffer_size) {
            grid_buffer[grid_buffer_idx*grid_buffer_size + count] = point_id;
        }
//        atomicExch(&grid_buffer[grid_buffer_idx], point_id);
    }
}


__global__ void voxel_sampling_idx_gpu_kernel(int batch_size, int center_num,
                                              int kernel_size,
                                              int grid_dim_w, int grid_dim_l, int grid_dim_h,
                                              float resolution_w, float resolution_l, float resolution_h,
                                              int grid_buffer_size, int output_pooling_size, bool with_rpn,
                                              const float* input_coors,
                                              const float* center_coors,
                                              int* center_accu_list,
                                              int* grid_buffer,
                                              int* grid_buffer_count,
                                              int* output_idx,
                                              int* output_idx_count,
                                              int* valid_idx) {

	const int kernel_num = kernel_size * kernel_size * kernel_size;
	const int half_kernel_size = (kernel_size - 1) / 2;
	const int half_kernel_num = kernel_size * kernel_size * half_kernel_size + \
                                kernel_size * half_kernel_size + \
                                half_kernel_size;
	const int search_kernel_size = kernel_size + 1;
	const int search_kernel_num = search_kernel_size * search_kernel_size * search_kernel_size;
    const int grid_dim_size = grid_dim_w * grid_dim_l * grid_dim_h;
	const float radius_x = 1.5 * resolution_w;
	const float radius_y = 1.5 * resolution_l;
	const float radius_z = 1.5 * resolution_h;
	const float r_x2 = radius_x * radius_x;
	const float r_y2 = radius_y * radius_y;
	const float r_z2 = radius_z * radius_z;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < center_num * search_kernel_num) {

        int center_id = thread_id / search_kernel_num;
        int search_grid_id = thread_id % search_kernel_num;
        int batch_id = get_batch_id(center_accu_list, batch_size, center_id);

        float center_coor_x = center_coors[center_id*3 + 0];
        float center_coor_y = center_coors[center_id*3 + 1];
        float center_coor_z = center_coors[center_id*3 + 2];
        int center_grid_coor_x = __float2int_rz(center_coor_x / resolution_w);
        int center_grid_coor_y = __float2int_rz(center_coor_y / resolution_l);
        int center_grid_coor_z = __float2int_rz(center_coor_z / resolution_h);


        int search_grid_x = search_grid_id / (search_kernel_size * search_kernel_size);
        int search_grid_y = search_grid_id % (search_kernel_size * search_kernel_size) / search_kernel_size;
        int search_grid_z = search_grid_id % search_kernel_size;

        int search_offset_x = -2 + round(center_coor_x / resolution_w - center_grid_coor_x) + search_grid_x;
        int search_offset_y = -2 + round(center_coor_y / resolution_l - center_grid_coor_y) + search_grid_y;
        int search_offset_z = -2 + round(center_coor_z / resolution_h - center_grid_coor_z) + search_grid_z;

        int target_grid_x = max(0, min(center_grid_coor_x + search_offset_x, grid_dim_w - 1));
        int target_grid_y = max(0, min(center_grid_coor_y + search_offset_y, grid_dim_l - 1));
        int target_grid_z = max(0, min(center_grid_coor_z + search_offset_z, grid_dim_h - 1));
        int target_grid_id = batch_id * grid_dim_size + target_grid_x * grid_dim_l * grid_dim_h + target_grid_y * grid_dim_h + target_grid_z;

        for (int p=0; p<grid_buffer_size; p++) {
            int point_id = grid_buffer[target_grid_id*grid_buffer_size + p];
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
                if (dx2 < r_x2 && dy2 < r_y2 && dz2 < r_z2) {
                    int kernel_coor_x = __float2int_rz(dx / resolution_w + 0.5 * fabsf(dx) / dx);
                    int kernel_coor_y = __float2int_rz(dy / resolution_l + 0.5 * fabsf(dy) / dy);
                    int kernel_coor_z = __float2int_rz(dz / resolution_h + 0.5 * fabsf(dz) / dz);
                    int voxel_coor = center_id * kernel_num + half_kernel_num + \
                                     kernel_size * kernel_size * kernel_coor_x + \
                                     kernel_size * kernel_coor_y + \
                                     kernel_coor_z;
                    int pooling_count = atomicAdd(&output_idx_count[voxel_coor], 1);
                    if (pooling_count < output_pooling_size) {
                        output_idx[voxel_coor*output_pooling_size + pooling_count] = point_id;
                        if (with_rpn)
                            atomicAdd(&valid_idx[center_id], 1);
                    }
                }
            }
        }
	}
}


void voxel_sampling_idx_gpu_launcher(int batch_size, int input_point_num,
                                     int center_num, int kernel_size,
                                     int grid_dim_w, int grid_dim_l, int grid_dim_h, std::vector<float> resolution,
                                     int grid_buffer_size, int output_pooling_size, bool with_rpn,
                                     const float* input_coors,
                                     const int* input_num_list,
                                     const float* center_coors,
                                     const int* center_num_list,
                                     int* input_accu_list,
                                     int* center_accu_list,
                                     int* grid_buffer,
                                     int* grid_buffer_count,
                                     int* output_idx,
                                     int* output_idx_count,
                                     int* valid_idx) {
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
                                                    output_pooling_size,
                                                    output_idx);

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, grid_buffer_init_gpu_kernel, 0, input_point_num);
    gridSize = (input_point_num + blockSize - 1) / blockSize;
    grid_buffer_init_gpu_kernel<<<gridSize, blockSize>>>(batch_size, input_point_num,
                                                         grid_dim_w, grid_dim_l, grid_dim_h,
                                                         resolution[0], resolution[1], resolution[2],
                                                         grid_buffer_size,
                                                         input_coors,
                                                         input_accu_list,
                                                         grid_buffer,
                                                         grid_buffer_count);

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, voxel_sampling_idx_gpu_kernel, 0, center_num * search_kernel_num);
    gridSize = (center_num * search_kernel_num + blockSize - 1) / blockSize;
    voxel_sampling_idx_gpu_kernel<<<gridSize, blockSize>>>(batch_size, center_num,
                                                           kernel_size,
                                                           grid_dim_w, grid_dim_l, grid_dim_h,
                                                           resolution[0], resolution[1], resolution[2],
                                                           grid_buffer_size, output_pooling_size, with_rpn,
                                                           input_coors,
                                                           center_coors,
                                                           center_accu_list,
                                                           grid_buffer,
                                                           grid_buffer_count,
                                                           output_idx,
                                                           output_idx_count,
                                                           valid_idx);
}
