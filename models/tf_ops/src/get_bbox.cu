/* Furthest point sampling GPU implementation
 * Author Zhaoyu SU
 * All Rights Reserved. Sep., 2019.
 * Happy Mid-Autumn Festival! :)
 */
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#define USECPSEC 1000000ULL

__global__ void get_bbox_gpu_kernel(int batch_size, int npoint, int nbbox, int bbox_attr, int diff_thres, int cls_thres, float expand_ratio,
                                   const float* roi_attrs,
                                   const float* gt_bbox,
                                   const int* input_num_list,
                                   int* input_accu_list,
                                   float* bbox,
                                   int* bbox_conf,
                                   int* bbox_diff) {
    if (batch_size * nbbox * bbox_attr <=0 || npoint <=0) {
//        printf("Get Bbox Logits Op exited unexpectedly.\n");
        return;
    }
    input_accu_list[0] = 0;
    for (int b=1; b<batch_size; b++) {
        input_accu_list[b] = input_accu_list[b-1] + input_num_list[b-1];
    }
    __syncthreads();
    for (int b=blockIdx.x; b<batch_size; b+=gridDim.x) {
        for (int i=threadIdx.x; i<input_num_list[b]; i+=blockDim.x) {
            bbox[input_accu_list[b]*7 + i*7 + 0] = 0.1;
            bbox[input_accu_list[b]*7 + i*7 + 1] = 0.1;
            bbox[input_accu_list[b]*7 + i*7 + 2] = 0.1;

            float roi_x = roi_attrs[input_accu_list[b]*7 + i*7 + 3];
            float roi_y = roi_attrs[input_accu_list[b]*7 + i*7 + 4];
            float roi_z = roi_attrs[input_accu_list[b]*7 + i*7 + 5];
            bbox_conf[input_accu_list[b] + i] = 0;
            bbox_diff[input_accu_list[b] + i] = -1;
            for (int j=0; j<nbbox; j++) {
            // [w, l, h, x, y, z, r, cls, diff_idx]
            //  0  1  2  3  4  5  6   7      8
                float bbox_w = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 0];
                float bbox_l = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 1];
                float bbox_h = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 2];
                float bbox_x = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 3];
                float bbox_y = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 4];
                float bbox_z = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 5];
                float bbox_r = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 6];
                float bbox_cls = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 7];
                float diff = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 8];

                if (bbox_l*bbox_h*bbox_w > 0) {
                    float rel_roi_x = roi_x - bbox_x;
                    float rel_roi_y = roi_y - bbox_y;
                    float rel_roi_z = roi_z - bbox_z;
                    float rot_rel_roi_x = rel_roi_x*cosf(bbox_r) + rel_roi_y*sinf(bbox_r);
                    float rot_rel_roi_y = -rel_roi_x*sinf(bbox_r) + rel_roi_y*cosf(bbox_r);
                    if (abs(rot_rel_roi_x) <= bbox_w * (1 + expand_ratio) / 2 &&
                        abs(rot_rel_roi_y) <= bbox_l * (1 + expand_ratio) / 2 &&
                        abs(rel_roi_z) <= bbox_h * (1 + expand_ratio) / 2) {

                        bbox[input_accu_list[b]*7 + i*7 + 0] = bbox_w;
                        bbox[input_accu_list[b]*7 + i*7 + 1] = bbox_l;
                        bbox[input_accu_list[b]*7 + i*7 + 2] = bbox_h;
                        bbox[input_accu_list[b]*7 + i*7 + 3] = bbox_x;
                        bbox[input_accu_list[b]*7 + i*7 + 4] = bbox_y;
                        bbox[input_accu_list[b]*7 + i*7 + 5] = bbox_z;
                        bbox[input_accu_list[b]*7 + i*7 + 6] = bbox_r;

//                        if (diff <= diff_thres && bbox_cls == 0) {
                        if (diff <= diff_thres && bbox_cls <= cls_thres) {
                            // Here we only take cars into consideration, while vans are excluded and give the foreground labels as -1 (ignored).
                            bbox_conf[input_accu_list[b] + i] = 1;
                            bbox_diff[input_accu_list[b] + i] = diff;
                        }
                        else {
                            bbox_conf[input_accu_list[b] + i] = -1;
                            bbox_diff[input_accu_list[b] + i] = -1;
                        }
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


void get_bbox_gpu_launcher(int batch_size, int npoint, int nbbox, int bbox_attr, int diff_thres, int cls_thres, float expand_ratio,
                          const float* roi_attrs,
                          const float* gt_bbox,
                          const int* input_num_list,
                          int* input_accu_list,
                          float* bbox,
                          int* bbox_conf,
                          int* bbox_diff) {
//    long long dt = dtime_usec(0);
    get_bbox_gpu_kernel<<<32,512>>>(batch_size, npoint, nbbox, bbox_attr, diff_thres, cls_thres, expand_ratio,
                                           roi_attrs,
                                           gt_bbox,
                                           input_num_list,
                                           input_accu_list,
                                           bbox,
                                           bbox_conf,
                                           bbox_diff);
//    dt = dtime_usec(dt);
//	std::cout << "Voxel Sample (forward) CUDA time: " << dt/(float)USECPSEC << "s" << std::endl;
}
