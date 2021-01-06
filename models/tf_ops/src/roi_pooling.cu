/* Voxel sampling GPU implementation
 * Author Zhaoyu SU
 * All Rights Reserved. Sep., 2019.
 */
#include <stdio.h>
#include <iostream>
#include <float.h>  // import FLT_EPSILON

__device__ int get_batch_id(int* accu_list, int batch_size, int id) {
    for (int b=0; b<batch_size-1; b++) {
        if (id >= accu_list[b]) {
            if(id < accu_list[b+1])
                return b;
        }
    }
    return batch_size - 1;
}


__global__ void output_init_gpu_kernel(int roi_num, int voxel_num, int channels, float padding_value,
                                       float* output_features,
                                       int* output_idx) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < roi_num * voxel_num) {
        int start_loc = thread_id * channels;
        for (int c=0; c<channels; c++) {
            output_features[start_loc + c] = padding_value;
            output_idx[start_loc + c] = -1;
        }
    }
}


__global__ void roi_pooling_register_gpu_kernel(int batch_size, int max_batch_input_point_num, int roi_num,
                                                int voxel_size, int pooling_size,
                                                const float* input_coors,
                                                const float* roi_attrs,
                                                const int* input_num_list,
                                                int* input_accu_list,
                                                int* roi_accu_list,
                                                int* temp_pool,
                                                int* temp_count) {
    const int voxel_num = voxel_size * voxel_size * voxel_size;
    const int half_voxel_size = (voxel_size - 1) / 2;
    const int center_offset = voxel_size * voxel_size * half_voxel_size + \
                              voxel_size * half_voxel_size + \
                              half_voxel_size;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < roi_num * max_batch_input_point_num) {
        int roi_id = thread_id / max_batch_input_point_num;
        int input_id = thread_id % max_batch_input_point_num;
        int batch_id = get_batch_id(roi_accu_list, batch_size, roi_id);

        if (input_id < input_num_list[batch_id]) {

            float roi_w = roi_attrs[roi_id*7 + 0];
            float roi_l = roi_attrs[roi_id*7 + 1];
            float roi_h = roi_attrs[roi_id*7 + 2];
            float roi_x = roi_attrs[roi_id*7 + 3];
            float roi_y = roi_attrs[roi_id*7 + 4];
            float roi_z = roi_attrs[roi_id*7 + 5];
            float roi_r = roi_attrs[roi_id*7 + 6];

            float grid_length_x = roi_w / voxel_size;
            float grid_length_y = roi_l / voxel_size;
            float grid_length_z = roi_h / voxel_size;

            int batch_id = get_batch_id(roi_accu_list, batch_size, roi_id);
            float point_x = input_coors[input_accu_list[batch_id]*3 + input_id*3 + 0];
            float point_y = input_coors[input_accu_list[batch_id]*3 + input_id*3 + 1];
            float point_z = input_coors[input_accu_list[batch_id]*3 + input_id*3 + 2];

            float rel_point_x = point_x - roi_x;
            float rel_point_y = point_y - roi_y;
            float rel_point_z = point_z - roi_z;

            float rot_rel_point_x = rel_point_x*cosf(-roi_r) - rel_point_y*sinf(-roi_r);
            float rot_rel_point_y = rel_point_x*sinf(-roi_r) + rel_point_y*cosf(-roi_r);

            if (abs(rot_rel_point_x)<roi_w / 2 && abs(rot_rel_point_y)<roi_l / 2 && abs(rel_point_z)<roi_h / 2) {
                int grid_coor_x = __float2int_rz(rot_rel_point_x / (grid_length_x + FLT_EPSILON) + 0.5 * fabsf(rot_rel_point_x) / rot_rel_point_x + FLT_EPSILON);
                int grid_coor_y = __float2int_rz(rot_rel_point_y / (grid_length_y + FLT_EPSILON) + 0.5 * fabsf(rot_rel_point_y) / rot_rel_point_y + FLT_EPSILON);
                int grid_coor_z = __float2int_rz(rel_point_z / (grid_length_z + FLT_EPSILON) + 0.5 * fabsf(rel_point_z) / rel_point_z + FLT_EPSILON);
                int voxel_coor = roi_id * voxel_num + center_offset + \
                                 voxel_size * voxel_size * grid_coor_x + \
                                 voxel_size * grid_coor_y + \
                                 grid_coor_z;

                int pool_count = atomicAdd(&temp_count[voxel_coor], 1);
                if (pool_count < pooling_size) {
                    temp_pool[voxel_coor * pooling_size + pool_count] = input_accu_list[batch_id] + input_id;
                }
            }
        }
    }
}

__global__ void roi_pooling_fill_gpu_kernel(int roi_num, int voxel_num, int channels, int pooling_size,
                                            const float* input_features,
                                            int* temp_count,
                                            int* temp_pool,
                                            float* output_features,
                                            int* output_idx) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < roi_num * voxel_num) {
        int pool_count = min(temp_count[thread_id], pooling_size);
        if (pool_count > 0) {
            for (int c=0; c<channels; c++) {
                float max_feature = -1e9;
                for (int p=0; p<pool_count; p++) {
                    int pool_id = thread_id * pooling_size + p;
                    int input_id = temp_pool[pool_id];
                    if (input_features[input_id * channels + c] > max_feature) {
                        max_feature = input_features[input_id * channels + c];
                        output_features[thread_id * channels + c] = max_feature;
                        output_idx[thread_id * channels + c] = input_id;
                    }
                }
            }
        }
    }
}


__global__ void roi_pooling_grad_gpu_kernel(int ncenters, int voxel_num, int channels,
                                             const int* output_idx,
                                             const float* output_features_grad,
                                             float* input_features_grad) {
    int center_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (center_id < ncenters) {
        for (int i=0; i<voxel_num; i++) {
            int voxel_coor = center_id*voxel_num + i;
            for (int c=0; c<channels; c++) {
                int point_id = output_idx[voxel_coor * channels + c];
                if (point_id != -1) {
                    atomicAdd(&input_features_grad[point_id*channels + c], output_features_grad[voxel_coor*channels + c]);
                }
            }
        }
    }
}

void roi_pooling_gpu_launcher(int batch_size, int input_point_num, int channels,
                              int roi_num, int voxel_size, int pooling_size, float padding_value,
                              const float* input_coors,
                              const float* input_features,
                              const float* roi_attrs,
                              const int* input_num_list,
                              const int* roi_num_list,
                              int* input_num_list_host,
                              int* input_accu_list,
                              int* roi_accu_list,
                              int* temp_pool,
                              int* temp_count,
                              float* output_features,
                              int* output_idx) {
    if (batch_size * channels <= 0) {
        printf("RoiPoolingOp ERROR: Invalid CUDA input dimensions.\n");
        return;
    }
    if (input_point_num <= 0)
        return;

    const int voxel_num = voxel_size * voxel_size * voxel_size;
    int max_batch_input_point_num = 0;
    for (int i=0; i<batch_size; i++)
        max_batch_input_point_num = max(max_batch_input_point_num, input_num_list_host[i]);


    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, output_init_gpu_kernel, 0, roi_num * voxel_num);
    gridSize = (roi_num * voxel_num + blockSize - 1) / blockSize;
    output_init_gpu_kernel<<<gridSize,blockSize>>>(roi_num, voxel_num, channels, padding_value,
                                                   output_features,
                                                   output_idx);

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, roi_pooling_register_gpu_kernel, 0, roi_num * max_batch_input_point_num);
    gridSize = (roi_num * max_batch_input_point_num + blockSize - 1) / blockSize;
    roi_pooling_register_gpu_kernel<<<gridSize,blockSize>>>(batch_size, max_batch_input_point_num, roi_num,
                                                            voxel_size, pooling_size,
                                                            input_coors,
                                                            roi_attrs,
                                                            input_num_list,
                                                            input_accu_list,
                                                            roi_accu_list,
                                                            temp_pool,
                                                            temp_count);

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, roi_pooling_fill_gpu_kernel, 0, roi_num * voxel_num);
    gridSize = (roi_num * voxel_num + blockSize - 1) / blockSize;
    roi_pooling_fill_gpu_kernel<<<gridSize,blockSize>>>(roi_num, voxel_num, channels, pooling_size,
                                                        input_features,
                                                        temp_count,
                                                        temp_pool,
                                                        output_features,
                                                        output_idx);
}


void roi_pooling_grad_gpu_launcher(int ncenters, int voxel_num, int channels,
                                   const int* output_idx,
                                   const float* output_features_grad,
                                   float* input_features_grad) {
    if (channels <= 0) {
        printf("RoiPoolingGradOp ERROR: Invalid CUDA input dimensions.\n");
        return;
    }
    if (ncenters == 0)
        return;
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, roi_pooling_grad_gpu_kernel, 0, ncenters);
    gridSize = (ncenters + blockSize - 1) / blockSize;

    roi_pooling_grad_gpu_kernel<<<gridSize, blockSize>>>(ncenters, voxel_num, channels,
                                             output_idx,
                                             output_features_grad,
                                             input_features_grad);
}