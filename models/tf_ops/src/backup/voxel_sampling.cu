/* Voxel sampling GPU implementation
 * Author Zhaoyu SU
 * All Rights Reserved. Sep., 2019.
 */
#include <stdio.h>
#include <iostream>
#include <float.h>

__device__ int binary_search(const long long* input_voxel_idx,
                             int start_id,
                             int stop_id,
                             long long target_voxel_id) {

    if (input_voxel_idx[start_id] > target_voxel_id || input_voxel_idx[stop_id] < target_voxel_id)
        return -1;
    while (start_id <= stop_id) {
        int m = start_id + (stop_id - start_id) / 2;
        if (input_voxel_idx[m] == target_voxel_id)
            return m;
        if (input_voxel_idx[m] < target_voxel_id)
            start_id = m + 1;
        else
            stop_id = m - 1;
    }
    return -1;
}

__device__ int get_batch_id(int* accu_list, int batch_size, int id) {
    for (int b=0; b<batch_size-1; b++) {
        if (id >= accu_list[b]) {
            if(id < accu_list[b+1])
                return b;
        }
    }
    return batch_size - 1;
}

__global__ void output_init_gpu_kernel(int batch_size, int ngrid, int channels,
                                       float padding, int kernel_number,
                                       const int* center_num_list,
                                       int* center_accu_list,
                                       float* output_features,
                                       int* output_idx) {
    int center_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (center_id < kernel_number) {
        int batch_id = get_batch_id(center_accu_list, batch_size, center_id);
        for (int i=0; i<ngrid; i++) {
            output_idx[center_id*ngrid + i] = -1;
            for (int c=0; c<channels; c++) {
                output_features[center_id*ngrid*channels + i*channels + c] = padding;
            }
        }
    }
}

__global__ void grid_buffer_init_gpu_kernel(int batch_size, int input_npoint, float resolution,
                                            int grid_w, int grid_l, int grid_h, int grid_size,
                                            const float* input_coors,
                                            int* input_accu_list,
                                            int* grid_buffer) {
    int point_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (point_id < input_npoint) {
        int grid_coor_w = (int)floor(input_coors[point_id*3 + 0] / resolution);
        int grid_coor_l = (int)floor(input_coors[point_id*3 + 1] / resolution);
        int grid_coor_h = (int)floor(input_coors[point_id*3 + 2] / resolution);
        int batch_id = get_batch_id(input_accu_list, batch_size, point_id);
        int grid_buffer_idx = batch_id * grid_size + grid_coor_w * grid_l * grid_h + grid_coor_l * grid_h + grid_coor_h;
        atomicExch(&grid_buffer[grid_buffer_idx], point_id);
    }
}


__global__ void voxel_sampling_gpu_kernel(int batch_size, int channels,
                                          int kernel_number, int kernel_size, int ngrid,
                                          int grid_w, int grid_l, int grid_h,
                                          float resolution,
                                          const float* input_coors,
                                          const float* input_features,
                                          const float* center_coors,
                                          int* center_accu_list,
                                          int* grid_buffer,
                                          float* output_features,
                                          int* output_idx) {

    const int grid_size = grid_w * grid_l * grid_h;
	const int half_kernel_size = (kernel_size - 1) / 2;
	const float radius = 1.5 * resolution;
	const float r2 = radius * radius;
	const int center_offset = kernel_size * kernel_size * half_kernel_size + \
                              kernel_size * half_kernel_size + \
                              half_kernel_size;

    int center_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (center_id < kernel_number) {
        int batch_id = get_batch_id(center_accu_list, batch_size, center_id);
        float x_c = center_coors[center_id*3 + 0];
        float y_c = center_coors[center_id*3 + 1];
        float z_c = center_coors[center_id*3 + 2];
        int grid_coor_w = __float2int_rz(x_c / resolution);
        int grid_coor_l = __float2int_rz(y_c / resolution);
        int grid_coor_h = __float2int_rz(z_c / resolution);
//	        long long grid_idx_c = grid_coor_h * grid_w * grid_l + grid_coor_l * grid_w + grid_coor_w;
        int grid_search_w_min, grid_search_w_max;
        int grid_search_l_min, grid_search_l_max;
        int grid_search_h_min, grid_search_h_max;
        if (grid_coor_w * resolution + 0.5 * resolution > x_c) {
            grid_search_w_min = grid_coor_w - 2;
            grid_search_w_max = grid_coor_w + 1;
        }else{
            grid_search_w_min = grid_coor_w - 1;
            grid_search_w_max = grid_coor_w + 2;
        }
        if (grid_coor_l * resolution + 0.5 * resolution > y_c) {
            grid_search_l_min = grid_coor_l - 2;
            grid_search_l_max = grid_coor_l + 1;
        }else{
            grid_search_l_min = grid_coor_l - 1;
            grid_search_l_max = grid_coor_l + 2;
        }
        if (grid_coor_h * resolution + 0.5 * resolution > z_c) {
            grid_search_h_min = grid_coor_h - 2;
            grid_search_h_max = grid_coor_h + 1;
        }else{
            grid_search_h_min = grid_coor_h - 1;
            grid_search_h_max = grid_coor_h + 2;
        }

        for (int w=max(0, grid_search_w_min); w<=min(grid_search_w_max, grid_w-1); w++) {
            for (int l=max(0, grid_search_l_min); l<=min(grid_search_l_max, grid_l-1); l++) {
                for (int h=max(0, grid_search_h_min); h<=min(grid_search_h_max, grid_h-1); h++) {
                    int target_grid_id = batch_id * grid_size + w * grid_l * grid_h + l * grid_h + h;
                    int point_id = grid_buffer[target_grid_id];
                    if (point_id>=0) {
                        float x_i = input_coors[point_id*3 +0];
                        float y_i = input_coors[point_id*3 +1];
                        float z_i = input_coors[point_id*3 +2];
                        float dx = x_i - x_c + FLT_EPSILON;
                        float dy = y_i - y_c + FLT_EPSILON;
                        float dz = z_i - z_c + FLT_EPSILON;
                        float dx2 = dx * dx;
                        float dy2 = dy * dy;
                        float dz2 = dz * dz;
                        if (dx2 < r2 && dy2 < r2 && dz2 < r2) {
                            int x_coor = __float2int_rz(dx / resolution + 0.5 * fabsf(dx) / dx);
                            int y_coor = __float2int_rz(dy / resolution + 0.5 * fabsf(dy) / dy);
                            int z_coor = __float2int_rz(dz / resolution + 0.5 * fabsf(dz) / dz);
                            int voxel_coor = center_id * ngrid + center_offset + \
                                             kernel_size * kernel_size * x_coor + \
                                             kernel_size * y_coor + \
                                             z_coor;
                            if (output_idx[voxel_coor] < 0) {
                                output_idx[voxel_coor] = point_id;
                                for (int c=0; c<channels; c++) {
                                    output_features[voxel_coor * channels + c] = input_features[point_id * channels + c];
                                }
                            }
                        }
                    }
                }
            }
        }
	}
}


__global__ void voxel_sampling_grad_gpu_kernel(int kernel_number, int ngrid, int channels,
                                               const int* output_idx,
                                               const float* output_features_grad,
                                               float* input_features_grad) {
    int center_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (center_id < kernel_number) {
        for (int i=0; i<ngrid; i++) {
            int voxel_coor = center_id*ngrid + i;
            int point_id = output_idx[voxel_coor];
            if (point_id >= 0) {
                for (int c=0; c<channels; c++) {
                    atomicAdd(&input_features_grad[point_id*channels + c], output_features_grad[voxel_coor*channels + c]);
                }
            }
        }
    }
}


void voxel_sampling_gpu_launcher(int batch_size, int input_npoint, int channels,
                                 int kernel_number, int kernel_size,
                                 int grid_w, int grid_l, int grid_h,
                                 float resolution, float padding,
                                 const float* input_coors,
                                 const float* input_features,
                                 const int* input_num_list,
                                 const float* center_coors,
                                 const int* center_num_list,
                                 int* input_accu_list,
                                 int* center_accu_list,
                                 int* grid_buffer,
                                 float* output_features,
                                 int* output_idx) {
    if (batch_size*input_npoint <=0 || channels * kernel_number <= 0) {
        printf("VoxelSampleOp ERROR: Invalid CUDA input dimensions.\n");
        return;
    }
    int ngrid = kernel_size * kernel_size * kernel_size;
    int grid_size = grid_w * grid_l * grid_h;

    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, output_init_gpu_kernel, 0, kernel_number);
    gridSize = (kernel_number + blockSize - 1) / blockSize;
    output_init_gpu_kernel<<<gridSize, blockSize>>>(batch_size, ngrid, channels,
                                                    padding, kernel_number,
                                                    center_num_list,
                                                    center_accu_list,
                                                    output_features,
                                                    output_idx);

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, grid_buffer_init_gpu_kernel, 0, input_npoint);
    gridSize = (input_npoint + blockSize - 1) / blockSize;
    grid_buffer_init_gpu_kernel<<<gridSize, blockSize>>>(batch_size, input_npoint, resolution,
                                                         grid_w, grid_l, grid_h, grid_size,
                                                         input_coors,
                                                         input_accu_list,
                                                         grid_buffer);

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, voxel_sampling_gpu_kernel, 0, kernel_number);
    gridSize = (kernel_number + blockSize - 1) / blockSize;
    voxel_sampling_gpu_kernel<<<gridSize, blockSize>>>(batch_size, channels,
                                                       kernel_number, kernel_size, ngrid,
                                                       grid_w, grid_l, grid_h,
                                                       resolution,
                                                       input_coors,
                                                       input_features,
                                                       center_coors,
                                                       center_accu_list,
                                                       grid_buffer,
                                                       output_features,
                                                       output_idx);
}


void voxel_sampling_grad_gpu_launcher(int kernel_number, int ngrid, int channels,
                                    const int* output_idx,
                                    const float* output_features_grad,
                                    float* input_features_grad) {
    if (kernel_number==0 || ngrid*channels == 0) {
        printf("VoxelSampleGradOp ERROR: Invalid CUDA input dimensions.\n");
        return;
    }
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, output_init_gpu_kernel, 0, kernel_number);
    gridSize = (kernel_number + blockSize - 1) / blockSize;
    voxel_sampling_grad_gpu_kernel<<<gridSize, blockSize>>>(kernel_number, ngrid, channels,
                                                            output_idx,
                                                            output_features_grad,
                                                            input_features_grad);
}