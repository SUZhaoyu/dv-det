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

__global__ void get_bev_gt_bbox_gpu_kernel(int batch_size, int npoint, int nbbox, int bbox_attr,
                                           int num_anchor, int anchor_attr,
                                           int diff_thres, int cls_thres, float expand_ratio,
                                           const float* input_coors,
                                           const float* label_bbox,
                                           const int* input_num_list,
                                           const float* anchor_param_list,
                                           int* input_accu_list,
                                           float* gt_bbox,
                                           int* gt_conf,
                                           int* label_idx) {
    if (batch_size * nbbox * bbox_attr <=0 || npoint <=0) {
//        printf("Get RoI Logits Op exited unexpectedly.\n");
        return;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        input_accu_list[0] = 0;
        for (int b=1; b<batch_size; b++) {
            input_accu_list[b] = input_accu_list[b-1] + input_num_list[b-1];
        }
    }
    __syncthreads();
//    printf("%d\n", input_accu_list[5]);
    for (int b=blockIdx.x; b<batch_size; b+=gridDim.x) {
        for (int i=threadIdx.x; i<input_num_list[b]; i+=blockDim.x) {
            for (int k=0; k<num_anchor; k++) {
                float point_x = input_coors[input_accu_list[b]*2 + i*2 + 0];
                float point_y = input_coors[input_accu_list[b]*2 + i*2 + 1];
                float point_z = anchor_param_list[k*anchor_attr + 3];

                gt_bbox[input_accu_list[b]*num_anchor*7 + i*num_anchor*7 + k*7 + 0] = 0.1;
                gt_bbox[input_accu_list[b]*num_anchor*7 + i*num_anchor*7 + k*7 + 1] = 0.1;
                gt_bbox[input_accu_list[b]*num_anchor*7 + i*num_anchor*7 + k*7 + 2] = 0.1;
                gt_conf[input_accu_list[b]*num_anchor + i*num_anchor + k] = 0;
                label_idx[input_accu_list[b]*num_anchor + i*num_anchor + k] = -1;

                for (int j=0; j<nbbox; j++) {
                // [w, l, h, x, y, z, r, cls, diff_idx]
                //  0  1  2  3  4  5  6   7      8
                    float bbox_w = label_bbox[b*nbbox*bbox_attr + j*bbox_attr + 0];
                    float bbox_l = label_bbox[b*nbbox*bbox_attr + j*bbox_attr + 1];
                    float bbox_h = label_bbox[b*nbbox*bbox_attr + j*bbox_attr + 2];
                    float bbox_x = label_bbox[b*nbbox*bbox_attr + j*bbox_attr + 3];
                    float bbox_y = label_bbox[b*nbbox*bbox_attr + j*bbox_attr + 4];
                    float bbox_z = label_bbox[b*nbbox*bbox_attr + j*bbox_attr + 5];
                    float bbox_r = label_bbox[b*nbbox*bbox_attr + j*bbox_attr + 6];
                    float bbox_cls = label_bbox[b*nbbox*bbox_attr + j*bbox_attr + 7];
                    float bbox_diff = label_bbox[b*nbbox*bbox_attr + j*bbox_attr + 8];
//                    printf("bbox:[%.2f,%.2f,%.2f], point:[%.2f,%.2f,%.2f]\n",bbox_w,bbox_l,bbox_h,point_x,point_y,point_z);
                    if (bbox_l*bbox_h*bbox_w > 0) {
                        float rel_point_x = point_x - bbox_x;
                        float rel_point_y = point_y - bbox_y;
                        float rel_point_z = point_z - bbox_z;
                        float rot_rel_point_x = rel_point_x*cosf(bbox_r) + rel_point_y*sinf(bbox_r);
                        float rot_rel_point_y = -rel_point_x*sinf(bbox_r) + rel_point_y*cosf(bbox_r);
                        if (abs(rot_rel_point_x)<=bbox_w * (1 + expand_ratio) / 2 &&
                            abs(rot_rel_point_y)<=bbox_l * (1 + expand_ratio) / 2 &&
                            abs(rel_point_z)<=bbox_h * (1 + expand_ratio) / 2) {

                            gt_bbox[input_accu_list[b]*num_anchor*7 + i*num_anchor*7 + k*7 + 0] = bbox_w;
                            gt_bbox[input_accu_list[b]*num_anchor*7 + i*num_anchor*7 + k*7 + 1] = bbox_l;
                            gt_bbox[input_accu_list[b]*num_anchor*7 + i*num_anchor*7 + k*7 + 2] = bbox_h;
                            gt_bbox[input_accu_list[b]*num_anchor*7 + i*num_anchor*7 + k*7 + 3] = bbox_x;
                            gt_bbox[input_accu_list[b]*num_anchor*7 + i*num_anchor*7 + k*7 + 4] = bbox_y;
                            gt_bbox[input_accu_list[b]*num_anchor*7 + i*num_anchor*7 + k*7 + 5] = bbox_z;
                            gt_bbox[input_accu_list[b]*num_anchor*7 + i*num_anchor*7 + k*7 + 6] = bbox_r;

                            if (bbox_diff <= diff_thres && bbox_cls <= cls_thres) {
                                // Here we only take cars into consideration, while vans are excluded and give the foreground labels as -1 (ignored).
                                // TODO: need to change the category class accordingly to the expected detection target.
                                gt_conf[input_accu_list[b]*num_anchor + i*num_anchor + k] = 1;
                                label_idx[input_accu_list[b]*num_anchor + i*num_anchor + k] = b * nbbox + j;
                            }else{
                                gt_conf[input_accu_list[b]*num_anchor + i*num_anchor + k] = -1;
                            }
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

void get_bev_gt_bbox_gpu_launcher(int batch_size, int npoint, int nbbox, int bbox_attr,
                                  int num_anchor, int anchor_attr,
                                  int diff_thres, int cls_thres, float expand_ratio,
                                  const float* input_coors,
                                  const float* label_bbox,
                                  const int* input_num_list,
                                  const float* anchor_param_list,
                                  int* input_accu_list,
                                  float* gt_bbox,
                                  int* gt_conf,
                                  int* label_idx) {
    long long dt = dtime_usec(0);
    get_bev_gt_bbox_gpu_kernel<<<32,512>>>(batch_size, npoint, nbbox, bbox_attr,
                                           num_anchor, anchor_attr,
                                           diff_thres, cls_thres, expand_ratio,
                                           input_coors,
                                           label_bbox,
                                           input_num_list,
                                           anchor_param_list,
                                           input_accu_list,
                                           gt_bbox,
                                           gt_conf,
                                           label_idx);
    dt = dtime_usec(dt);
//	std::cout << "Voxel Sample (forward) CUDA time: " << dt/(float)USECPSEC << "s" << std::endl;
}
