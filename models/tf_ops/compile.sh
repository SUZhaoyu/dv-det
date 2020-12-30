if [ ! -d ./build ]; then
  mkdir build
fi

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# nvcc -std=c++11 -c -o build/get_roi_bbox.cu.o src/get_roi_bbox.cu ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 -shared src/get_roi_bbox.cpp build/get_roi_bbox.cu.o -o build/get_roi_bbox.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -lcudart -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
# nvcc -std=c++11 -c -o build/get_bbox.cu.o src/get_bbox.cu ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 -shared src/get_bbox.cpp build/get_bbox.cu.o -o build/get_bbox.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -lcudart -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
# nvcc -std=c++11 -c -o build/grid_sampling.cu.o src/grid_sampling.cu ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 -shared src/grid_sampling.cpp build/grid_sampling.cu.o -o build/grid_sampling.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -lcudart -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
# nvcc -std=c++11 -c -o build/voxel_sampling.cu.o src/voxel_sampling.cu ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 -shared src/voxel_sampling.cpp build/voxel_sampling.cu.o -o build/voxel_sampling.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -lcudart -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
# nvcc -std=c++11 -c -o build/roi_pooling.cu.o src/roi_pooling.cu ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 -shared src/roi_pooling.cpp build/roi_pooling.cu.o -o build/roi_pooling.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -lcudart -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
# g++ -std=c++11 -shared src/roi_filter.cpp -o build/roi_filter.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -lcudart -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
# nvcc -std=c++11 -c -o build/voxel2col.cu.o src/voxel2col.cu ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 -shared src/voxel2col.cpp build/voxel2col.cu.o -o build/voxel2col.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -lcudart -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
# nvcc -std=c++11 -c -o build/roi_logits_to_attrs.cu.o src/roi_logits_to_attrs.cu ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 -shared src/roi_logits_to_attrs.cpp build/roi_logits_to_attrs.cu.o -o build/roi_logits_to_attrs.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -lcudart -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
# nvcc -std=c++11 -c -o build/bbox_logits_to_attrs.cu.o src/bbox_logits_to_attrs.cu ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 -shared src/bbox_logits_to_attrs.cpp build/bbox_logits_to_attrs.cu.o -o build/bbox_logits_to_attrs.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -lcudart -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include

nvcc src/get_roi_bbox.cu -o build/get_roi_bbox.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 src/get_roi_bbox.cpp build/get_roi_bbox.cu.o -o build/get_roi_bbox.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include

nvcc src/get_bbox.cu -o build/get_bbox.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 src/get_bbox.cpp build/get_bbox.cu.o -o build/get_bbox.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include

nvcc src/grid_sampling.cu -o build/grid_sampling.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 src/grid_sampling.cpp build/grid_sampling.cu.o -o build/grid_sampling.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include

nvcc src/voxel_sampling.cu -o build/voxel_sampling.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 src/voxel_sampling.cpp build/voxel_sampling.cu.o -o build/voxel_sampling.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include

nvcc src/roi_pooling.cu -o build/roi_pooling.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 src/roi_pooling.cpp build/roi_pooling.cu.o -o build/roi_pooling.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include

g++ -std=c++11 src/roi_filter.cpp -o build/roi_filter.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include

nvcc src/voxel2col.cu -o build/voxel2col.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 src/voxel2col.cpp build/voxel2col.cu.o -o build/voxel2col.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include

nvcc src/roi_logits_to_attrs.cu -o build/roi_logits_to_attrs.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 src/roi_logits_to_attrs.cpp build/roi_logits_to_attrs.cu.o -o build/roi_logits_to_attrs.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include

nvcc src/bbox_logits_to_attrs.cu -o build/bbox_logits_to_attrs.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 src/bbox_logits_to_attrs.cpp build/bbox_logits_to_attrs.cu.o -o build/bbox_logits_to_attrs.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
