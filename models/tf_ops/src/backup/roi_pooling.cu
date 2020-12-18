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

__global__ void roi_pooling_gpu_kernel(int batch_size, int input_npoint, int channels,
                                       int kernel_number, int voxel_size, int pooling_size, float padding_value,
                                       const float* input_coors,
                                       const float* input_features,
                                       const float* roi_attrs,
                                       const int* input_num_list,
                                       const int* roi_num_list,
                                       int* input_accu_list,
                                       int* roi_accu_list,
                                       int* temp_pool,
                                       int* temp_count,
                                       float* output_features,
                                       int* output_idx) {
//    const float PI = 3.1415927;
    const int ngrid = voxel_size * voxel_size * voxel_size;
    const int half_voxel_size = (voxel_size - 1) / 2;
    const int center_offset = voxel_size * voxel_size * half_voxel_size + \
                              voxel_size * half_voxel_size + \
                              half_voxel_size;
    int attr_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (attr_id < kernel_number) {
        for (int i=0; i<ngrid; i++) {
            for (int c=0; c<channels; c++) {
                output_features[attr_id*ngrid + i*ngrid + c] = padding_value;
                output_idx[attr_id*ngrid + i*ngrid + c] = -1;
            }
        }
    }
    __syncthreads();

    if (attr_id < kernel_number) {
        float roi_w = roi_attrs[attr_id*7 + 0];
        float roi_l = roi_attrs[attr_id*7 + 1];
        float roi_h = roi_attrs[attr_id*7 + 2];
        float roi_x = roi_attrs[attr_id*7 + 3];
        float roi_y = roi_attrs[attr_id*7 + 4];
        float roi_z = roi_attrs[attr_id*7 + 5];
        float roi_r = roi_attrs[attr_id*7 + 6];

        float grid_length_x = roi_w / voxel_size;
        float grid_length_y = roi_l / voxel_size;
        float grid_length_z = roi_h / voxel_size;

        int batch_id = get_batch_id(roi_accu_list, batch_size, attr_id);
        for (int i=0; i<input_num_list[batch_id]; i++) {
            float point_x = input_coors[input_accu_list[batch_id]*3 + i*3 + 0];
            float point_y = input_coors[input_accu_list[batch_id]*3 + i*3 + 1];
            float point_z = input_coors[input_accu_list[batch_id]*3 + i*3 + 2];

            float rel_point_x = point_x - roi_x;
            float rel_point_y = point_y - roi_y;
            float rel_point_z = point_z - roi_z;

            float rot_rel_point_x = rel_point_x*cosf(roi_r) + rel_point_y*sinf(roi_r);
            float rot_rel_point_y = -rel_point_x*sinf(roi_r) + rel_point_y*cosf(roi_r);

            if (abs(rot_rel_point_x)<roi_w / 2 && abs(rot_rel_point_y)<roi_l / 2 && abs(rel_point_z)<roi_h / 2) {
                int grid_coor_x = __float2int_rz(rot_rel_point_x / (grid_length_x + FLT_EPSILON) + 0.5 * fabsf(rot_rel_point_x) / rot_rel_point_x + FLT_EPSILON);
                int grid_coor_y = __float2int_rz(rot_rel_point_y / (grid_length_y + FLT_EPSILON) + 0.5 * fabsf(rot_rel_point_y) / rot_rel_point_y + FLT_EPSILON);
                int grid_coor_z = __float2int_rz(rel_point_z / (grid_length_z + FLT_EPSILON) + 0.5 * fabsf(rel_point_z) / rel_point_z + FLT_EPSILON);
                int voxel_coor = attr_id * ngrid + center_offset + \
                                 voxel_size * voxel_size * grid_coor_x + \
                                 voxel_size * grid_coor_y + \
                                 grid_coor_z;
                int pool_count = temp_count[voxel_coor];
                if (pool_count < pooling_size) {
                    temp_pool[voxel_coor * pooling_size + pool_count] = input_accu_list[batch_id] + i;
                    atomicAdd(&temp_count[voxel_coor], 1);
                }
            }
        }
        for (int i=0; i<ngrid; i++) {
            int voxel_coor = attr_id*ngrid + i;
            int pool_count = temp_count[voxel_coor];
            if (pool_count > 0) {
                for (int c=0; c<channels; c++) {
                    float max = -1e9;
                    for (int p=0; p<pool_count; p++) {
                        int pool_idx = voxel_coor * pooling_size + p;
                        int input_idx = temp_pool[pool_idx];
                        if (input_features[input_idx * channels + c] > max) {
                            max = input_features[input_idx * channels + c];
                            output_features[voxel_coor * channels + c] = max;
                            output_idx[voxel_coor * channels + c] = input_idx;
                        }
                    }
                }
            }
        }
    }
}





__global__ void roi_pooling_grad_gpu_kernel(int ncenters, int ngrid, int channels,
                                             const int* output_idx,
                                             const float* output_features_grad,
                                             float* input_features_grad) {
    int center_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (center_id < ncenters) {
        for (int i=0; i<ngrid; i++) {
            int voxel_coor = center_id*ngrid + i;
            for (int c=0; c<channels; c++) {
                int point_id = output_idx[voxel_coor * channels + c];
                if (point_id != -1) {
                    atomicAdd(&input_features_grad[point_id*channels + c], output_features_grad[voxel_coor*channels + c]);
                }
            }
        }
    }
}

void roi_pooling_gpu_launcher(int batch_size, int input_npoint, int channels,
                              int kernel_number, int voxel_size, int pooling_size, float padding_value,
                              const float* input_coors,
                              const float* input_features,
                              const float* roi_attrs,
                              const int* input_num_list,
                              const int* roi_num_list,
                              int* input_accu_list,
                              int* roi_accu_list,
                              int* temp_pool,
                              int* temp_count,
                              float* output_features,
                              int* output_idx) {
    if (batch_size*input_npoint <=0 || channels <= 0) {
        printf("RoiPoolingOp ERROR: Invalid CUDA input dimensions.\n");
        return;
    }
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, roi_pooling_gpu_kernel, 0, kernel_number);
    gridSize = (kernel_number + blockSize - 1) / blockSize;

    roi_pooling_gpu_kernel<<<gridSize,blockSize>>>(batch_size, input_npoint, channels,
                                       kernel_number, voxel_size, pooling_size, padding_value,
                                       input_coors,
                                       input_features,
                                       roi_attrs,
                                       input_num_list,
                                       roi_num_list,
                                       input_accu_list,
                                       roi_accu_list,
                                       temp_pool,
                                       temp_count,
                                       output_features,
                                       output_idx);
}


void roi_pooling_grad_gpu_launcher(int ncenters, int ngrid, int channels,
                                   const int* output_idx,
                                   const float* output_features_grad,
                                   float* input_features_grad) {
    if (ncenters * channels <= 0) {
        printf("RoiPoolingGradOp ERROR: Invalid CUDA input dimensions.\n");
        return;
    }
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, roi_pooling_grad_gpu_kernel, 0, ncenters);
    gridSize = (ncenters + blockSize - 1) / blockSize;

    roi_pooling_grad_gpu_kernel<<<gridSize, blockSize>>>(ncenters, ngrid, channels,
                                             output_idx,
                                             output_features_grad,
                                             input_features_grad);
}