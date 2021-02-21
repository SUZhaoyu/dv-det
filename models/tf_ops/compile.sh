if [ ! -d ./build ]; then
  mkdir build
fi

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )



#nvcc src/get_roi_bbox.cu -o build/get_roi_bbox.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 src/get_roi_bbox.cpp build/get_roi_bbox.cu.o -o build/get_roi_bbox.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
#nvcc src/get_bbox.cu -o build/get_bbox.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 src/get_bbox.cpp build/get_bbox.cu.o -o build/get_bbox.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
#nvcc src/grid_sampling.cu -o build/grid_sampling.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 src/grid_sampling.cpp build/grid_sampling.cu.o -o build/grid_sampling.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
#nvcc src/voxel_sampling.cu -o build/voxel_sampling.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 src/voxel_sampling.cpp build/voxel_sampling.cu.o -o build/voxel_sampling.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
#nvcc src/roi_pooling.cu -o build/roi_pooling.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 src/roi_pooling.cpp build/roi_pooling.cu.o -o build/roi_pooling.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
#g++ -std=c++11 src/roi_filter.cpp -o build/roi_filter.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
#nvcc src/voxel2col.cu -o build/voxel2col.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 src/voxel2col.cpp build/voxel2col.cu.o -o build/voxel2col.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
#nvcc src/roi_logits_to_attrs.cu -o build/roi_logits_to_attrs.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 src/roi_logits_to_attrs.cpp build/roi_logits_to_attrs.cu.o -o build/roi_logits_to_attrs.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
#nvcc src/bbox_logits_to_attrs.cu -o build/bbox_logits_to_attrs.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 src/bbox_logits_to_attrs.cpp build/bbox_logits_to_attrs.cu.o -o build/bbox_logits_to_attrs.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
#nvcc src/voxel_sampling_binary.cu -o build/voxel_sampling_binary.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 src/voxel_sampling_binary.cpp build/voxel_sampling_binary.cu.o -o build/voxel_sampling_binary.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
#nvcc src/unique.cu -o build/unique.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 src/unique.cpp build/unique.cu.o -o build/unique.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
#nvcc src/nms.cu -o build/nms.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 src/nms.cpp build/nms.cu.o -o build/nms.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
#nvcc src/la_roi_pooling.cu -o build/la_roi_pooling.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 src/la_roi_pooling.cpp build/la_roi_pooling.cu.o -o build/la_roi_pooling.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
#nvcc src/la_roi_pooling_fast.cu -o build/la_roi_pooling_fast.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 src/la_roi_pooling_fast.cpp build/la_roi_pooling_fast.cu.o -o build/la_roi_pooling_fast.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
#nvcc src/voxel_sampling_idx.cu -o build/voxel_sampling_idx.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 src/voxel_sampling_idx.cpp build/voxel_sampling_idx.cu.o -o build/voxel_sampling_idx.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
#nvcc src/voxel_sampling_feature.cu -o build/voxel_sampling_feature.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 src/voxel_sampling_feature.cpp build/voxel_sampling_feature.cu.o -o build/voxel_sampling_feature.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
nvcc src/voxel_sampling_idx_binary.cu -o build/voxel_sampling_idx_binary.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 src/voxel_sampling_idx_binary.cpp build/voxel_sampling_idx_binary.cu.o -o build/voxel_sampling_idx_binary.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
