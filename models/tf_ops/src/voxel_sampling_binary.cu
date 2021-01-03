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

__global__ void voxel_sampling_binary_gpu_kernel(int batch_size, int input_npoint, int channels,
                                        int output_ncenter, int kernel_size,
                                        float dim_w, float dim_l, float dim_h,
                                        float resolution, float padding,
                                        const float* input_coors,
                                        const float* input_features,
                                        const long long* input_voxel_idx,
                                        const int* input_num_list,
                                        const float* center_coors,
                                        const int* center_num_list,
                                        int* input_accu_list,
                                        int* center_accu_list,
                                        float* output_features,
                                        int* output_idx) {

    if (batch_size*input_npoint <=0 || channels * output_ncenter <= 0) {
//        printf("Voxel sample Op exited unexpectedly.\n");
        return;
    }
	const int half_kernel_size = (kernel_size - 1) / 2;
	const float radius = 1.5 * resolution;
	const float r2 = radius * radius;
	const int ngrid = kernel_size * kernel_size * kernel_size;
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
	        for (int j=0; j<ngrid; j++) {
	            output_idx[center_accu_list[b]*ngrid + i*ngrid + j] = -1;
	            for (int u=0; u<channels; u++) {
	                output_features[center_accu_list[b]*ngrid*channels + i*ngrid*channels + j*channels + u] = padding;
	            }
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
                        int id = binary_search(input_voxel_idx,
                                               input_accu_list[b],
                                               input_accu_list[b] + input_num_list[b] - 1,
                                               target_grid_id);
//                        if (id > 100000)
//                            printf("************VoxelSamplingBinaryOpId: %d\n", id);
                        if (id>=0) {
                            float x_i = input_coors[id*3 + 0];
                            float y_i = input_coors[id*3 + 1];
                            float z_i = input_coors[id*3 + 2];
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
                                int voxel_coor = center_accu_list[b] * ngrid + i * ngrid + center_offset + \
                                                 kernel_size * kernel_size * x_coor + \
                                                 kernel_size * y_coor + \
                                                 z_coor;
                                if (output_idx[voxel_coor] < 0) {
                                    output_idx[voxel_coor] = id;
                                    for (int c=0; c<channels; c++) {
                                        output_features[voxel_coor * channels + c] = input_features[id*channels + c];
                                    }
                                }
                            }
                        }
	                }
	            }
	        }
	    }
	}
}


__global__ void voxel_sampling_binary_grad_gpu_kernel(int output_ncenter, int ngrid, int channels,
                                                     const int* output_idx,
                                                     const float* output_features_grad,
                                                     float* input_features_grad) {
    int batch_size = __float2int_ru((float)output_ncenter / blockDim.x);
    if (output_ncenter==0 || ngrid*channels == 0) {
        printf("Voxel sample grad Op exited unexpectedly.\n");
        return;
    }
    for (int b=blockIdx.x; b<batch_size; b+=gridDim.x) {
//        for (int i=threadIdx.x; b*blockDim.x + i < ncenters; i+=blockDim.x) {}
        int center_id = b*blockDim.x + threadIdx.x;
        if (center_id < output_ncenter) {
            for (int i=0; i<ngrid; i++) {
                int voxel_coor = center_id*ngrid + i;
                int id = output_idx[voxel_coor];
                if (id != -1) {
//                    if (id > 1000000)
//                        printf("************VoxelSamplingBinaryOpId: %d@[voxel_coor=%d, center_id=%d, i=%d, output_ncenter=%d]\n", id, voxel_coor, center_id, i, output_ncenter);
                     for (int c=0; c<channels; c++) {
                        atomicAdd(&input_features_grad[id*channels + c], output_features_grad[voxel_coor*channels + c]);
                     }
                }
            }
        }
    }
}

long long dtime_usec(unsigned long long start){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}


void voxel_sampling_binary_gpu_launcher(int batch_size, int input_npoint, int channels,
                               int output_ncenter, int kernel_size,
                               float dim_w, float dim_l, float dim_h,
                               float resolution, float padding,
                               const float* input_coors,
                               const float* input_features,
                               const long long* input_voxel_idx,
                               const int* input_num_list,
                               const float* center_coors,
                               const int* center_num_list,
                               int* input_accu_list,
                               int* center_accu_list,
                               float* output_features,
                               int* output_idx) {
    long long dt = dtime_usec(0);
    voxel_sampling_binary_gpu_kernel<<<32,512>>>(batch_size, input_npoint, channels,
                                        output_ncenter, kernel_size,
                                        dim_w, dim_l, dim_h,
                                        resolution, padding,
                                        input_coors,
                                        input_features,
                                        input_voxel_idx,
                                        input_num_list,
                                        center_coors,
                                        center_num_list,
                                        input_accu_list,
                                        center_accu_list,
                                        output_features,
                                        output_idx);
//    dt = dtime_usec(dt);
//	std::cout << "Voxel Sample (forward) CUDA time: " << dt/(float)USECPSEC << "s" << std::endl;


}


void voxel_sampling_binary_grad_gpu_launcher(int output_ncenter, int ngrid, int channels,
                                             const int* output_idx,
                                            const float* output_features_grad,
                                             float* input_features_grad) {
    voxel_sampling_binary_grad_gpu_kernel<<<32, 512>>>(output_ncenter, ngrid, channels,
                                                       output_idx,
                                                       output_features_grad,
                                                       input_features_grad);
}