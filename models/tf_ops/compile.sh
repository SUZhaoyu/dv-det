if [ ! -d ./build ]; then
  mkdir build
fi

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# nvcc src/unique.cu -o build/unique.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 src/unique.cpp build/unique.cu.o -o build/unique.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include

# nvcc src/voxel_sample.cu -o build/voxel_sample.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 src/voxel_sample.cpp build/voxel_sample.cu.o -o build/voxel_sample.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include

# g++ -std=c++11 src/kernel_conv.cpp -o build/kernel_conv.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include

# nvcc src/get_roi_bbox.cu -o build/get_roi_bbox.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 src/get_roi_bbox.cpp build/get_roi_bbox.cu.o -o build/get_roi_bbox.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
# nvcc src/get_bbox.cu -o build/get_bbox.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 src/get_bbox.cpp build/get_bbox.cu.o -o build/get_bbox.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include

# nvcc src/grid_sampling.cu -o build/grid_sampling.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 src/grid_sampling.cpp build/grid_sampling.cu.o -o build/grid_sampling.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include

# nvcc src/voxel_sampling.cu -o build/voxel_sampling.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 src/voxel_sampling.cpp build/voxel_sampling.cu.o -o build/voxel_sampling.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include

# nvcc src/roi_pooling.cu -o build/roi_pooling.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 src/roi_pooling.cpp build/roi_pooling.cu.o -o build/roi_pooling.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include

# g++ -std=c++11 src/roi_filter.cpp -o build/roi_filter.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include

nvcc src/dense_conv.cu -o build/dense_conv.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 src/dense_conv.cpp build/dense_conv.cu.o -o build/dense_conv.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
