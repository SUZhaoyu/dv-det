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


__global__ void output_init_gpu_kernel(int roi_num, int voxel_num, int channels, int pooling_size, float padding_value,
                                       float* output_features,
                                       int* output_idx) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < roi_num * voxel_num) {
        for (int c=0; c<channels; c++) {
            output_features[thread_id * channels + c] = padding_value;
        }
        for (int p=0; p<pooling_size; p++) {
            output_idx[thread_id * pooling_size + p] = -1;
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
                                                int* temp_count,
                                                int* output_idx,
                                                float* output_weight) {
    const int voxel_num = voxel_size * voxel_size * voxel_size;
    const int half_voxel_size = (voxel_size - 1) / 2;
//    const int center_offset = voxel_size * voxel_size * half_voxel_size + \
//                              voxel_size * half_voxel_size + \
//                              half_voxel_size;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < roi_num * voxel_num * max_batch_input_point_num) {
        int roi_id = thread_id / (max_batch_input_point_num * voxel_num);
        int voxel_id = thread_id / max_batch_input_point_num % voxel_num;
        int input_id = thread_id % max_batch_input_point_num;
        int batch_id = get_batch_id(roi_accu_list, batch_size, roi_id);
//        if (thread_id % 100 == 0) {
//            printf("%d, %d, %d, %d, %d\n", thread_id, roi_id, voxel_id, input_id, batch_id);
//            printf("%d, %d\n", batch_size, roi_id);
//        }
        if (input_id < input_num_list[batch_id]) {

            float roi_w = roi_attrs[roi_id*7 + 0];
            float roi_l = roi_attrs[roi_id*7 + 1];
            float roi_h = roi_attrs[roi_id*7 + 2];
            float roi_x = roi_attrs[roi_id*7 + 3];
            float roi_y = roi_attrs[roi_id*7 + 4];
            float roi_z = roi_attrs[roi_id*7 + 5];
            float roi_r = roi_attrs[roi_id*7 + 6];

            int grid_coor_x = voxel_id / (voxel_size * voxel_size);
            int grid_coor_y = (voxel_id - grid_coor_x * voxel_size * voxel_size) / voxel_size;
            int grid_coor_z = voxel_id % voxel_size;
            grid_coor_x -= half_voxel_size;
            grid_coor_y -= half_voxel_size;
            grid_coor_z -= half_voxel_size;
//            printf("%d, %d, %d\n", grid_coor_x, grid_coor_y, grid_coor_z);

            float grid_length_x = roi_w / voxel_size;
            float grid_length_y = roi_l / voxel_size;
            float grid_length_z = roi_h / voxel_size;
            float radius = max(grid_length_x, grid_length_y);

            float rel_grid_x = grid_coor_x * grid_length_x;
            float rel_grid_y = grid_coor_y * grid_length_y;
            float rel_grid_z = grid_coor_z * grid_length_z;

            float rot_rel_grid_x = rel_grid_x*cosf(roi_r) - rel_grid_y*sinf(roi_r);
            float rot_rel_grid_y = rel_grid_x*sinf(roi_r) + rel_grid_y*cosf(roi_r);

            float act_grid_x = rot_rel_grid_x + roi_x;
            float act_grid_y = rot_rel_grid_y + roi_y;
            float act_grid_z = rel_grid_z + roi_z;

            int batch_id = get_batch_id(roi_accu_list, batch_size, roi_id);
            float point_x = input_coors[input_accu_list[batch_id]*3 + input_id*3 + 0];
            float point_y = input_coors[input_accu_list[batch_id]*3 + input_id*3 + 1];
            float point_z = input_coors[input_accu_list[batch_id]*3 + input_id*3 + 2];
            float dist_2 = pow(point_x - act_grid_x, 2.) + pow(point_y - act_grid_y, 2.) + pow(point_z - act_grid_z, 2.);


            if (dist_2 < pow(radius, 2.)) {
//                printf("Yes\n");
                float dist = sqrt(dist_2);
                float weight = 2.71828 / expf(dist / radius);
//                float weight = 1.;
                int voxel_coor = roi_id * voxel_num + voxel_id;
                int pool_count = atomicAdd(&temp_count[voxel_coor], 1);
                if (pool_count < pooling_size) {
                    output_idx[voxel_coor * pooling_size + pool_count] = input_accu_list[batch_id] + input_id;
                    output_weight[voxel_coor * pooling_size + pool_count] = weight;
                }
            }
        }
    }
}

__global__ void roi_pooling_fill_gpu_kernel(int roi_num, int voxel_num, int channels, int pooling_size,
                                            const float* input_features,
                                            int* temp_count,
                                            float* output_features,
                                            int* output_idx,
                                            float* output_weight) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < roi_num * voxel_num) {
        int pool_count = min(temp_count[thread_id], pooling_size);
        if (pool_count > 0) {
            float weight_sum = 0;
            int pool_id;
            for (int p=0; p<pool_count; p++) {
                pool_id = thread_id * pooling_size + p;
                weight_sum += output_weight[pool_id];
            }
            for (int p=0; p<pool_count; p++) {
                pool_id = thread_id * pooling_size + p;
                int input_id = output_idx[pool_id];
                float weight = output_weight[pool_id] * output_weight[pool_id] / weight_sum;
//                printf("%d, %d, %f\n", input_id, pool_count, weight);
                output_weight[pool_id] = weight;
                for (int c=0; c < channels; c++) {
                    output_features[thread_id * channels + c] += input_features[input_id * channels + c] * weight;
//                    printf("%d, %d\n", thread_id, input_id);
                }
            }
        }
    }
}


__global__ void roi_pooling_grad_gpu_kernel(int roi_num, int voxel_num, int channels, int pooling_size,
                                             const int* output_idx,
                                             const float* output_weight,
                                             const float* output_features_grad,
                                             float* input_features_grad) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < roi_num * voxel_num) {
        for (int p=0; p<pooling_size; p++) {
            int point_id = output_idx[thread_id * pooling_size + p];
            float weight = output_weight[thread_id * pooling_size + p];
            if (point_id >= 0) {
//                printf("%d, %d, %f\n", point_id, thread_id, output_features_grad[thread_id*channels] * weight);
                for (int c=0; c<channels; c++) {
                    atomicAdd(&input_features_grad[point_id*channels + c], output_features_grad[thread_id*channels + c] * weight);
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
                              int* temp_count,
                              float* output_features,
                              int* output_idx,
                              float* output_weight) {
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
    output_init_gpu_kernel<<<gridSize,blockSize>>>(roi_num, voxel_num, channels, pooling_size, padding_value,
                                                   output_features,
                                                   output_idx);

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, roi_pooling_register_gpu_kernel, 0, roi_num * voxel_num * max_batch_input_point_num);
    gridSize = (roi_num * voxel_num * max_batch_input_point_num + blockSize - 1) / blockSize;
//    printf("<<<%d, %d>>>\n", gridSize, blockSize);
    roi_pooling_register_gpu_kernel<<<gridSize,blockSize>>>(batch_size, max_batch_input_point_num, roi_num,
                                                            voxel_size, pooling_size,
                                                            input_coors,
                                                            roi_attrs,
                                                            input_num_list,
                                                            input_accu_list,
                                                            roi_accu_list,
                                                            temp_count,
                                                            output_idx,
                                                            output_weight);

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, roi_pooling_fill_gpu_kernel, 0, roi_num * voxel_num);
    gridSize = (roi_num * voxel_num + blockSize - 1) / blockSize;
    roi_pooling_fill_gpu_kernel<<<gridSize,blockSize>>>(roi_num, voxel_num, channels, pooling_size,
                                                        input_features,
                                                        temp_count,
                                                        output_features,
                                                        output_idx,
                                                        output_weight);
}


void roi_pooling_grad_gpu_launcher(int roi_num, int voxel_num, int channels, int pooling_size,
                                   const int* output_idx,
                                   const float* output_weight,
                                   const float* output_features_grad,
                                   float* input_features_grad) {
    if (channels <= 0) {
        printf("RoiPoolingGradOp ERROR: Invalid CUsDA input dimensions.\n");
        return;
    }
    if (roi_num == 0)
        return;
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, roi_pooling_grad_gpu_kernel, 0, roi_num * voxel_num);
    gridSize = (roi_num * voxel_num + blockSize - 1) / blockSize;

    roi_pooling_grad_gpu_kernel<<<gridSize, blockSize>>>(roi_num, voxel_num, channels, pooling_size,
                                             output_idx,
                                             output_weight,
                                             output_features_grad,
                                             input_features_grad);
}