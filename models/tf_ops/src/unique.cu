#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <iostream>



int unique_gpu_launcher(long long* input_voxel_ids_temp,
                        int* input_point_ids_temp,
                        int input_npoint) {
//  key: voxel_id;
//  value: point_ids.
    thrust::sort_by_key(thrust::device, input_voxel_ids_temp, input_voxel_ids_temp+input_npoint, input_point_ids_temp);
    thrust::pair<long long*,int*> new_end;
    new_end = thrust::unique_by_key(thrust::device, input_voxel_ids_temp, input_voxel_ids_temp+input_npoint, input_point_ids_temp, thrust::equal_to<long long>());
//    printf("%d\n", new_end.second - input_point_ids_temp);
    int unique_count = new_end.second - input_point_ids_temp;
    return unique_count;
//    cout<<*unique_count<<endl;
}