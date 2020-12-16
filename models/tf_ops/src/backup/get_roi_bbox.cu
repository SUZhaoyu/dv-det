/* Furthest point sampling GPU implementation
 * Author Zhaoyu SU
 * All Rights Reserved. Sep., 2019.
 * Happy Mid-Autumn Festival! :)
 */
#include <stdio.h>
#include <iostream>

__device__ int get_batch_id(int* accu_list, int batch_size, int id) {
    for (int b=0; b<batch_size-1; b++) {
        if (id >= accu_list[b]) {
            if(id < accu_list[b+1])
                return b;
        }
    }
    return batch_size - 1;
}


__global__ void get_roi_bbox_gpu_kernel(int batch_size, int npoint, int nbbox, int bbox_attr,
                                        int diff_thres, float expand_ratio,
                                        const float* input_coors,
                                        const float* gt_bbox,
                                        const int* input_num_list,
                                        const float* anchor_size,
                                        int* input_accu_list,
                                        float* roi_bbox,
                                        int* roi_conf,
                                        int* roi_diff) {

    const float anchor_diag = sqrtf(anchor_size[0]*anchor_size[0] + anchor_size[1]*anchor_size[1]);
    int point_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (point_id < npoint) {
        // Initialize all the bbox with 0.1*0.1*0.1 dimension for lateral loss calculation.
        roi_bbox[point_id * 7 + 0] = 0.1;
        roi_bbox[point_id * 7 + 1] = 0.1;
        roi_bbox[point_id * 7 + 2] = 0.1;
        roi_conf[point_id] = 0;
        roi_diff[point_id] = -1;
        float point_x = input_coors[point_id * 3 + 0];
        float point_y = input_coors[point_id * 3 + 1];
        float point_z = input_coors[point_id * 3 + 2];
        int batch_id = get_batch_id(input_accu_list, batch_size, point_id);
        for (int i=0; i<nbbox; i++) {
            float bbox_w = gt_bbox[batch_id*nbbox*bbox_attr + i*bbox_attr + 0];
            float bbox_l = gt_bbox[batch_id*nbbox*bbox_attr + i*bbox_attr + 1];
            float bbox_h = gt_bbox[batch_id*nbbox*bbox_attr + i*bbox_attr + 2];
            float bbox_x = gt_bbox[batch_id*nbbox*bbox_attr + i*bbox_attr + 3];
            float bbox_y = gt_bbox[batch_id*nbbox*bbox_attr + i*bbox_attr + 4];
            float bbox_z = gt_bbox[batch_id*nbbox*bbox_attr + i*bbox_attr + 5];
            float bbox_r = gt_bbox[batch_id*nbbox*bbox_attr + i*bbox_attr + 6];
            float bbox_cls = gt_bbox[batch_id*nbbox*bbox_attr + i*bbox_attr + 7];
            float bbox_diff = gt_bbox[batch_id*nbbox*bbox_attr + i*bbox_attr + 8];
            if (bbox_l*bbox_h*bbox_w > 0) {
                float rel_point_x = point_x - bbox_x;
                float rel_point_y = point_y - bbox_y;
                float rel_point_z = point_z - bbox_z;
                float rot_rel_point_x = rel_point_x*cosf(bbox_r) + rel_point_y*sinf(bbox_r);
                float rot_rel_point_y = -rel_point_x*sinf(bbox_r) + rel_point_y*cosf(bbox_r);
                if (abs(rot_rel_point_x)<=bbox_w * (1 + expand_ratio) / 2 &&
                    abs(rot_rel_point_y)<=bbox_l * (1 + expand_ratio) / 2 &&
                    abs(rel_point_z)<=bbox_h * (1 + expand_ratio) / 2) {

                    roi_bbox[point_id * 7 + 0] = bbox_w;
                    roi_bbox[point_id * 7 + 1] = bbox_l;
                    roi_bbox[point_id * 7 + 2] = bbox_h;
                    roi_bbox[point_id * 7 + 3] = bbox_x;
                    roi_bbox[point_id * 7 + 4] = bbox_y;
                    roi_bbox[point_id * 7 + 5] = bbox_z;
                    roi_bbox[point_id * 7 + 6] = bbox_r;

                    if (bbox_diff <= diff_thres && bbox_cls == 0) {
                        // Here we only take cars into consideration, while vans are excluded and give the foreground labels as -1 (ignored).
                        // TODO: need to change the category class accordingly to the expected detection target.
                        roi_conf[point_id] = 1;
                        roi_diff[point_id] = bbox_diff;
                    }else{ // ignore
                        roi_conf[point_id] = -1;
                        roi_diff[point_id] = -1;
                    }
                }
            }
        }
    }
}


void get_roi_bbox_gpu_launcher(int batch_size, int npoint, int nbbox, int bbox_attr,
                               int diff_thres, float expand_ratio,
                               const float* input_coors,
                               const float* gt_bbox,
                               const int* input_num_list,
                               const float* anchor_size,
                               int* input_accu_list,
                               float* roi_bbox,
                               int* roi_conf,
                               int* roi_diff) {
    if (batch_size * nbbox * bbox_attr <=0 || npoint <=0) {
        printf("GetRoIBbox Op exited unexpectedly due to invalid CUDA input dimension.\n");
        return;
    }
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, get_roi_bbox_gpu_kernel, 0, npoint);
    gridSize = (npoint + blockSize - 1) / blockSize;
    get_roi_bbox_gpu_kernel<<<gridSize, blockSize>>>(batch_size, npoint, nbbox, bbox_attr,
                                                     diff_thres, expand_ratio,
                                                     input_coors,
                                                     gt_bbox,
                                                     input_num_list,
                                                     anchor_size,
                                                     input_accu_list,
                                                     roi_bbox,
                                                     roi_conf,
                                                     roi_diff);
}
