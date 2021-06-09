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


__global__ void grid_buffer_init_gpu_kernel(int batch_size, int input_point_num,
                                            float grid_buffer_resolution, int grid_buffer_size,
                                            int grid_dim_w, int grid_dim_l, int grid_dim_h,
                                            const float* input_coors,
                                            int* input_accu_list,
                                            int* grid_buffer,
                                            int* grid_buffer_count) {
    const int grid_dim_size = grid_dim_w * grid_dim_h * grid_dim_l;
    int point_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (point_id < input_point_num) {
        int grid_coor_x = __float2int_rz(input_coors[point_id*3 + 0] / grid_buffer_resolution);
        int grid_coor_y = __float2int_rz(input_coors[point_id*3 + 1] / grid_buffer_resolution);
        int grid_coor_z = __float2int_rz(input_coors[point_id*3 + 2] / grid_buffer_resolution);
        grid_coor_x = max(0, min(grid_coor_x, grid_dim_w - 1));
        grid_coor_y = max(0, min(grid_coor_y, grid_dim_l - 1));
        grid_coor_z = max(0, min(grid_coor_z, grid_dim_h - 1));
        int batch_id = get_batch_id(input_accu_list, batch_size, point_id);
        int grid_buffer_idx = batch_id * grid_dim_size + grid_coor_x * grid_dim_l * grid_dim_h + grid_coor_y * grid_dim_h + grid_coor_z;
        int count = atomicAdd(&grid_buffer_count[grid_buffer_idx], 1);
        if (count < grid_buffer_size)
            grid_buffer[grid_buffer_idx * grid_buffer_size + count] = point_id;
    }
}


__global__ void roi_pooling_register_gpu_kernel(int batch_size, int roi_num,
                                                int voxel_size, int pooling_size,
                                                float grid_buffer_resolution, int grid_buffer_size,
                                                float offset_w, float offset_l, float offset_h,
                                                int grid_dim_w, int grid_dim_l, int grid_dim_h,
                                                const float* input_coors,
                                                const float* roi_attrs,
                                                const int* input_num_lists,
                                                int* input_accu_list,
                                                int* roi_accu_list,
                                                int* temp_count,
                                                int* grid_buffer,
                                                int* grid_buffer_count,
                                                int* output_idx,
                                                float* output_weight) {
    const int voxel_num = voxel_size * voxel_size * voxel_size;
    const int grid_dim_size = grid_dim_w * grid_dim_l * grid_dim_h;
    const int half_voxel_size = (voxel_size - voxel_size % 2) / 2;
//    const int center_offset = voxel_size * voxel_size * half_voxel_size + \
//                              voxel_size * half_voxel_size + \
//                              half_voxel_size;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < roi_num * voxel_num) {
        int roi_id = thread_id / voxel_num;
        int voxel_id = thread_id % voxel_num;
        int voxel_coor = roi_id * voxel_num + voxel_id;
        int batch_id = get_batch_id(roi_accu_list, batch_size, roi_id);

        float roi_w = roi_attrs[roi_id*7 + 0];
        float roi_l = roi_attrs[roi_id*7 + 1];
        float roi_h = roi_attrs[roi_id*7 + 2];
        float roi_x = roi_attrs[roi_id*7 + 3] + offset_w;
        float roi_y = roi_attrs[roi_id*7 + 4] + offset_l;
        float roi_z = roi_attrs[roi_id*7 + 5] + offset_h;
        float roi_r = roi_attrs[roi_id*7 + 6];

//            printf("%d, %d, %d\n", grid_coor_x, grid_coor_y, grid_coor_z);

        float roi_grid_length_x = roi_w / voxel_size;
        float roi_grid_length_y = roi_l / voxel_size;
        float roi_grid_length_z = roi_h / voxel_size;
//

        int roi_grid_coor_x = voxel_id / (voxel_size * voxel_size);
        int roi_grid_coor_y = (voxel_id - roi_grid_coor_x * voxel_size * voxel_size) / voxel_size;
        int roi_grid_coor_z = voxel_id % voxel_size;

        roi_grid_coor_x -= half_voxel_size;
        roi_grid_coor_y -= half_voxel_size;
        roi_grid_coor_z -= half_voxel_size;

        float rel_roi_grid_x = (roi_grid_coor_x + 0.5 * (1 - voxel_size % 2)) * roi_grid_length_x;
        float rel_roi_grid_y = (roi_grid_coor_y + 0.5 * (1 - voxel_size % 2)) * roi_grid_length_y;
        float rel_roi_grid_z = (roi_grid_coor_z + 0.5 * (1 - voxel_size % 2)) * roi_grid_length_z;


        float rot_rel_roi_grid_x = rel_roi_grid_x*cosf(roi_r) - rel_roi_grid_y*sinf(roi_r);
        float rot_rel_roi_grid_y = rel_roi_grid_x*sinf(roi_r) + rel_roi_grid_y*cosf(roi_r);

        float act_roi_grid_x = rot_rel_roi_grid_x + roi_x;
        float act_roi_grid_y = rot_rel_roi_grid_y + roi_y;
        float act_roi_grid_z = rel_roi_grid_z + roi_z;

        int buffer_grid_coor_x = __float2int_rz(act_roi_grid_x / grid_buffer_resolution);
        int buffer_grid_coor_y = __float2int_rz(act_roi_grid_y / grid_buffer_resolution);
        int buffer_grid_coor_z = __float2int_rz(act_roi_grid_z / grid_buffer_resolution);

        int begin_buffer_grid_coor_x = max(0, buffer_grid_coor_x - 1);
        int begin_buffer_grid_coor_y = max(0, buffer_grid_coor_y - 1);
        int begin_buffer_grid_coor_z = max(0, buffer_grid_coor_z - 1);
        int stop_buffer_grid_coor_x = min(buffer_grid_coor_x + 1, grid_dim_w - 1);
        int stop_buffer_grid_coor_y = min(buffer_grid_coor_y + 1, grid_dim_l - 1);
        int stop_buffer_grid_coor_z = min(buffer_grid_coor_z + 1, grid_dim_h - 1);

        for (int x=begin_buffer_grid_coor_x; x<=stop_buffer_grid_coor_x; x++) {
            for (int y=begin_buffer_grid_coor_y; y<=stop_buffer_grid_coor_y; y++) {
                for (int z=begin_buffer_grid_coor_z; z<=stop_buffer_grid_coor_z; z++) {
                    if (temp_count[voxel_coor] >= pooling_size)
                        break;
                    int search_grid_id = batch_id * grid_dim_size + x * grid_dim_l * grid_dim_h + y * grid_dim_h + z;
                    int valid_buffer_count = min(grid_buffer_count[search_grid_id], grid_buffer_size);
//                    printf("grid buffer count = %d\n", grid_buffer_count[search_grid_id]);
                    for (int i=0; i<valid_buffer_count; i++) {
                        int point_id = grid_buffer[search_grid_id * grid_buffer_size + i];
                        float point_x = input_coors[point_id*3 + 0];
                        float point_y = input_coors[point_id*3 + 1];
                        float point_z = input_coors[point_id*3 + 2];

                        float rel_point_x = point_x - act_roi_grid_x;
                        float rel_point_y = point_y - act_roi_grid_y;
                        float rel_point_z = point_z - act_roi_grid_z;

                        float rel_rot_point_x = rel_point_x*cosf(-roi_r) - rel_point_y*sinf(-roi_r);
                        float rel_rot_point_y = rel_point_x*sinf(-roi_r) + rel_point_y*cosf(-roi_r);

//                        float dist_2 = pow(point_x - act_roi_grid_x, 2.) + pow(point_y - act_roi_grid_y, 2.) + pow(point_z - act_roi_grid_z, 2.);
//                        if (dist_2 < pow(radius, 2.)) {
                        if (abs(rel_rot_point_x) <= roi_grid_length_x * 1.2 &&
                            abs(rel_rot_point_y) <= roi_grid_length_y * 1.2 &&
                            abs(rel_point_z) <= roi_grid_length_z * 1.2) {
            //                printf("Yes\n");
                            float dist_2 = pow(point_x - act_roi_grid_x, 2.) + pow(point_y - act_roi_grid_y, 2.) + pow(point_z - act_roi_grid_z, 2.);
                            float dist = sqrtf(dist_2);
                            float radius = max(max(roi_grid_length_x, roi_grid_length_y), roi_grid_length_z) / 2.;
                            float weight = __expf(-dist / radius);
//                            float weight = 1.;

                            int pool_count = temp_count[voxel_coor];
                            if (pool_count < pooling_size) {
                                output_idx[voxel_coor * pooling_size + pool_count] = point_id;
                                output_weight[voxel_coor * pooling_size + pool_count] = weight;
                                temp_count[voxel_coor] += 1;
                            } else {
                                break;
                            }
                        }
                    }
                }
            }
        }
//        printf("Voxel Pooling = %d\n", temp_count[voxel_coor]);
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
//                weight_sum += 1;
            }
            for (int p=0; p<pool_count; p++) {
                int pool_id = thread_id * pooling_size + p;
                int input_id = output_idx[pool_id];
                float weight = output_weight[pool_id] * output_weight[pool_id] / weight_sum;
//                float weight = output_weight[pool_id];
//                printf("%d, %d, %f\n", input_id, pool_count, weight);
                output_weight[pool_id] = weight;
//                if (thread_id == 0)
//                    printf("Forward: %f\n", weight);
//                float temp_max = -1e6;
                for (int c=0; c < channels; c++) {
                    output_features[thread_id * channels + c] += input_features[input_id * channels + c] * weight;
//                    float features = input_features[input_id * channels + c] * weight;
//                    if (features > temp_max) {
//                        output_features[thread_id * channels + c] = features;
//                        temp_max = features;
//                    }
////                    printf("%d, %d\n", thread_id, input_id);
                }
            }
        }
    }
}

__global__ void roi_pooling_grad_gpu_kernel(int ncenters, int voxel_num, int channels, int pooling_size,
                                            const int* output_idx,
                                            const float* output_weight,
                                            const float* output_features_grad,
                                            float* input_features_grad) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < ncenters * voxel_num * pooling_size) {
        int pooling_id = thread_id % pooling_size;
        int center_id = thread_id / (pooling_size * voxel_num);
        int voxel_id = thread_id / pooling_size % voxel_num;

        int voxel_coor = center_id*voxel_num + voxel_id;
        int point_id = output_idx[voxel_coor * pooling_size + pooling_id];
        float point_weight = output_weight[voxel_coor * pooling_size + pooling_id];
//        if (thread_id == 0)
//        printf("Center_id=%d, voxel_id=%d, pooling_id=%d, point_id=%d\n", center_id, voxel_id, pooling_id, point_id);

        for (int c=0; c<channels; c++) {
            if (point_id >= 0) {
                atomicAdd(&input_features_grad[point_id*channels + c], output_features_grad[voxel_coor*channels + c] * point_weight);
            }
        }
    }
}


void roi_pooling_gpu_launcher(int batch_size, int input_point_num, int channels,
                              int roi_num, int voxel_size, int pooling_size, float padding_value,
                              float grid_buffer_resolution, int grid_buffer_size,
                              int grid_buffer_dim_w, int grid_buffer_dim_l, int grid_buffer_dim_h,
                              float offset_w, float offset_l, float offset_h,
                              const float* input_coors,
                              const float* input_features,
                              const float* roi_attrs,
                              const int* input_num_list,
                              const int* roi_num_list,
                              int* input_num_list_host,
                              int* input_accu_list,
                              int* roi_accu_list,
                              int* temp_count,
                              int* temp_grid_buffer,
                              int* temp_grid_buffer_count,
                              float* output_features,
                              int* output_idx,
                              float* output_weight) {

    if (batch_size * channels <= 0 || input_point_num <= 0) {
        printf("RoiPoolingFastOp ERROR: Invalid CUDA input dimensions.\n");
        return;
    }
    if (roi_num <= 0) {
//        printf("RoiPoolingFastOp WARNING: No RoIs were found for the current batch.\n");
        return;
    }
//    printf("RoI Num: %d\n", roi_num);
    const int voxel_num = voxel_size * voxel_size * voxel_size;


    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, output_init_gpu_kernel, 0, roi_num * voxel_num);
    gridSize = (roi_num * voxel_num + blockSize - 1) / blockSize;
    output_init_gpu_kernel<<<gridSize,blockSize>>>(roi_num, voxel_num, channels, pooling_size, padding_value,
                                                   output_features,
                                                   output_idx);

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, grid_buffer_init_gpu_kernel, 0, input_point_num);
    gridSize = (input_point_num + blockSize - 1) / blockSize;
    grid_buffer_init_gpu_kernel<<<gridSize,blockSize>>>(batch_size, input_point_num,
                                                        grid_buffer_resolution, grid_buffer_size,
                                                        grid_buffer_dim_w, grid_buffer_dim_l, grid_buffer_dim_h,
                                                        input_coors,
                                                        input_accu_list,
                                                        temp_grid_buffer,
                                                        temp_grid_buffer_count);

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, roi_pooling_register_gpu_kernel, 0, roi_num * voxel_num);
    gridSize = (roi_num * voxel_num + blockSize - 1) / blockSize;
    roi_pooling_register_gpu_kernel<<<gridSize,blockSize>>>(batch_size, roi_num,
                                                            voxel_size, pooling_size,
                                                            grid_buffer_resolution, grid_buffer_size,
                                                            offset_w, offset_l, offset_h,
                                                            grid_buffer_dim_w, grid_buffer_dim_l, grid_buffer_dim_h,
                                                            input_coors,
                                                            roi_attrs,
                                                            input_num_list,
                                                            input_accu_list,
                                                            roi_accu_list,
                                                            temp_count,
                                                            temp_grid_buffer,
                                                            temp_grid_buffer_count,
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

void roi_pooling_grad_gpu_launcher(int ncenters, int voxel_num, int channels, int pooling_size,
                                   const int* output_idx,
                                   const float* output_weight,
                                   const float* output_features_grad,
                                   float* input_features_grad) {
    if (channels * voxel_num <= 0) {
        printf("RoiPoolingGradOp ERROR: Invalid CUDA input dimensions.\n");
        return;
    }
    if (ncenters == 0)
        return;
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, roi_pooling_grad_gpu_kernel, 0, ncenters * voxel_num * pooling_size);
    gridSize = (ncenters * voxel_num * pooling_size + blockSize - 1) / blockSize;

    roi_pooling_grad_gpu_kernel<<<gridSize, blockSize>>>(ncenters, voxel_num, channels, pooling_size,
                                                         output_idx,
                                                         output_weight,
                                                         output_features_grad,
                                                         input_features_grad);
}