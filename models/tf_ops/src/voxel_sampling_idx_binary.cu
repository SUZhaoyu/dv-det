/* Voxel sampling GPU implementation
 * Author Zhaoyu SU
 * All Rights Reserved. Sep., 2019.
 */
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#define USECPSEC 1000000ULL

__device__ int binary_search(const long long* input_voxel_idx,
                             int start_id,
                             int stop_id,
                             long long target_voxel_id) {
//    if (threadIdx.x==0)
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

//__device__ int start_loc_search(const long long* input_voxel_idx,
//                                int grid_buffer_size,
//                                int start_id, int loc) {
//    long long input_idx = input_voxel_idx[loc];
//    long long query_idx = input_idx;
//    int start_loc = loc;
//    int count = 0;
//    while(query_idx == input_idx && start_loc > start_id && count < grid_buffer_size) {
//        query_idx = input_voxel_idx[start_loc];
//        start_loc -= 1;
//        count += 1;
//    }
//    if (query_idx == input_idx)
//        start_loc += 1;
//    return start_loc;
//}
//
//__device__ int stop_loc_search(const long long* input_voxel_idx,
//                               int grid_buffer_size,
//                               int stop_id, int loc) {
//    long long input_idx = input_voxel_idx[loc];
//    long long query_idx = input_idx;
//    int stop_loc = loc;
//    int count = 0;
//    while(query_idx == input_idx && stop_loc < stop_id && count < grid_buffer_size) {
//        query_idx = input_voxel_idx[stop_loc];
//        stop_loc += 1;
//        count += 1;
//    }
//    if (query_idx == input_idx)
//        stop_loc -= 1;
//    return stop_loc;
//}


__global__ void voxel_sampling_idx_binary_gpu_kernel(int batch_size, int input_npoint,
                                                     int center_num, int kernel_size,
                                                     float dim_w, float dim_l, float dim_h,
                                                     float resolution,
                                                     int grid_buffer_size, int output_pooling_size,
                                                     const float* input_coors,
                                                     const long long* input_voxel_idx,
                                                     const int* input_num_list,
                                                     const float* center_coors,
                                                     const int* center_num_list,
                                                     int* input_accu_list,
                                                     int* center_accu_list,
                                                     int* output_idx,
                                                     int* output_idx_count) {

    if (batch_size*input_npoint <=0 || center_num <= 0) {
//        printf("Voxel sample Op exited unexpectedly.\n");
        return;
    }
    printf("here\n");
	const int half_kernel_size = (kernel_size - 1) / 2;
	const float radius = 1.5 * resolution;
	const float r2 = radius * radius;
	const int kernel_num = kernel_size * kernel_size * kernel_size;
	const int center_offset = kernel_size * kernel_size * half_kernel_size + \
                              kernel_size * half_kernel_size + \
                              half_kernel_size;
	const float EPS = 1e-6;
	int grid_w = dim_w / resolution;
	int grid_l = dim_l / resolution;
	int grid_h = dim_h / resolution;
//	printf("%f, %f, %f\n", dim_w, dim_l, dim_h);

	for (int b=blockIdx.x; b<batch_size; b+=gridDim.x) {
	    for (int i=threadIdx.x; i<center_num_list[b]; i+=blockDim.x) {
	        for (int j=0; j<kernel_num; j++) {
	            output_idx[center_accu_list[b]*kernel_num + i*kernel_num + j] = -1;
	        }
	    }
	    __syncthreads();



	    for (int i=threadIdx.x; i<center_num_list[b]; i+=blockDim.x) {
	        float x_c = center_coors[center_accu_list[b]*3 + i*3 + 0];
	        float y_c = center_coors[center_accu_list[b]*3 + i*3 + 1];
	        float z_c = center_coors[center_accu_list[b]*3 + i*3 + 2];
	        int grid_coor_w = x_c / resolution;
	        int grid_coor_l = y_c / resolution;
	        int grid_coor_h = z_c / resolution;
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
                        long long target_grid_id = h * grid_w * grid_l + l * grid_w + w;
                        int batch_start_id = input_accu_list[b];
                        int batch_stop_id = input_accu_list[b] + input_num_list[b] - 1;
                        int id = binary_search(input_voxel_idx,
                                               batch_start_id,
                                               batch_stop_id,
                                               target_grid_id);
//                        if (id > 100000)
//                            printf("************VoxelSamplingBinaryOpId: %d\n", id);
                        if (id>=0) {
                            printf("%d, %d\n", batch_start_id, batch_stop_id);
                            int i = id;
//                            int start_id = start_loc_search(input_voxel_idx, grid_buffer_size, batch_start_id, id);
//                            int stop_id = stop_loc_search(input_voxel_idx, grid_buffer_size, batch_stop_id, id);
//                            printf("%d, %d\n", start_id, stop_id);
//                            for (int i=start_id; i<stop_id && i<grid_buffer_size; i++) {
                            float x_i = input_coors[i*3 + 0];
                            float y_i = input_coors[i*3 + 1];
                            float z_i = input_coors[i*3 + 2];
                            float dx = x_i - x_c + EPS;
                            float dy = y_i - y_c + EPS;
                            float dz = z_i - z_c + EPS;
                            float dx2 = dx * dx;
                            float dy2 = dy * dy;
                            float dz2 = dz * dz;
                            if (dx2 < r2 && dy2 < r2 && dz2 < r2) {
                                int x_coor = __float2int_rz(dx / resolution + 0.5 * fabsf(dx) / dx);
                                int y_coor = __float2int_rz(dy / resolution + 0.5 * fabsf(dy) / dy);
                                int z_coor = __float2int_rz(dz / resolution + 0.5 * fabsf(dz) / dz);
                                int voxel_coor = center_accu_list[b] * kernel_num + i * kernel_num + center_offset + \
                                                 kernel_size * kernel_size * x_coor + \
                                                 kernel_size * y_coor + \
                                                 z_coor;
                                int pooling_count = atomicAdd(&output_idx_count[voxel_coor], 1);
                                if (pooling_count < output_pooling_size) {
                                    output_idx[voxel_coor*output_pooling_size + pooling_count] = i;
                                }
                            }
//                            }
                        }
	                }
	            }
	        }
	    }
	}
}


void voxel_sampling_idx_binary_gpu_launcher(int batch_size, int input_npoint,
                                            int center_num, int kernel_size,
                                            float dim_w, float dim_l, float dim_h,
                                            float resolution,
                                            int grid_buffer_size, int output_pooling_size,
                                            const float* input_coors,
                                            const long long* input_voxel_idx,
                                            const int* input_num_list,
                                            const float* center_coors,
                                            const int* center_num_list,
                                            int* input_accu_list,
                                            int* center_accu_list,
                                            int* output_idx,
                                            int* output_idx_count) {

    voxel_sampling_idx_binary_gpu_kernel<<<32,1024>>>(batch_size, input_npoint,
                                                      center_num, kernel_size,
                                                      dim_w, dim_l, dim_h,
                                                      resolution,
                                                      grid_buffer_size, output_pooling_size,
                                                      input_coors,
                                                      input_voxel_idx,
                                                      input_num_list,
                                                      center_coors,
                                                      center_num_list,
                                                      input_accu_list,
                                                      center_accu_list,
                                                      output_idx,
                                                      output_idx_count);
}
