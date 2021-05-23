#!/usr/bin/env bash
#exe_file="train_stage1_anchor_x2.py"
exe_file="$1/$2.py"
#exe_file="train_stage1.py"
pkill -f -9 $exe_file

HOME="dv-det"
checkpoints="ckpt-$1"

root_gx4="/home/tan/tony"
root_gx8="/home/tan/tony"
ip_gx4="192.168.69.54"
ip_gx8="192.168.69.58"

rsync -avz -W -e ssh --progress \
                     --exclude='*.pyc' \
                     --exclude='build' \
                     --exclude='eval' \
                     --exclude='img_*.npy' \
                     --exclude='*_testing.npy' \
                     --exclude='waymo_npy' \
                     --exclude='ckpt-arxiv' \
                     --exclude='.nv' \
                     $root_gx4/$HOME tan@$ip_gx8:$root_gx8

echo "Pushing Completed!"


conda_env_gx4="/home/tan/anaconda3/envs/detection/bin/python"
home_dir_gx4="$root_gx4/$HOME"
exe_dir_gx4="$home_dir_gx4/train/$exe_file"
#
conda_env_gx8="/home/tan/anaconda3/envs/detection/bin/python"
home_dir_gx8="$root_gx8/$HOME"
exe_dir_gx8="$home_dir_gx8/train/$exe_file"

if [ ! -d "$home_dir_gx4/$checkpoints" ]; then
  mkdir -p "$home_dir_gx4/$checkpoints"
fi

echo "Input the task name:"
read task_name
if ((${#task_name} == 0)); then
	echo "ERROR: Task name can not be empty.";
	exit;
fi

log_dir="$home_dir_gx4/$checkpoints/$task_name"
actual_task_name=$task_name

if [ -d "$log_dir" ]; then
	dir_exists=true
	while $dir_exists; do
		echo "The checkpoint dir: $log_dir already exists, choose another name, "
		echo "or type 'y' to delete the existing dir and create a new one."
		read task_name
		if ((${#task_name} == 0)); then
			echo "ERROR: Task name can not be empty.";
			exit;
		elif [ "$task_name" = "y" ]; then
			task_name=$actual_task_name
			log_dir="$home_dir_gx4/$checkpoints/$task_name"
			dir_exists=false
			rm -rf $log_dir
			echo "WARNINIG: $log_dir has been cleaned."
		else
			log_dir="$home_dir_gx4/$checkpoints/$task_name"
			if [ -d "$log_dir" ]; then
				dir_exists=true;
			else
				dir_exists=false;
			fi
		fi
	done
fi
mkdir $log_dir

mpirun -np 8 \
       -H $ip_gx4:8\
       -bind-to none -map-by slot \
       -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
       -mca pml ob1 -mca btl ^openib \
       $conda_env_gx4 $exe_dir_gx4 --log_dir $log_dir :\
       -np 8 \
       -H $ip_gx8:8\
       -bind-to none -map-by slot \
       -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
       -mca pml ob1 -mca btl ^openib \
       $conda_env_gx8 $exe_dir_gx8

#mpirun -np 1 \
#       -H $ip_gx4:1\
#       -bind-to none -map-by slot \
#       -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
#       --mca btl ^vader --oversubscribe --mca pml ob1  \
#       $conda_env_gx4 $exe_dir_gx4 --log_dir $log_dir :\
#       -np 1 \
#       -H $ip_gx8:1\
#       -bind-to none -map-by slot \
#       -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
#       --mca btl ^vader --oversubscribe --mca pml ob1  \
#       $conda_env_gx8 $exe_dir_gx8

#horovodrun -np 8 -H $ip_gx4:8 $conda_env_gx4 $exe_dir_gx4 --log_dir $log_dir

# /usr/mpi/gcc/openmpi-4.0.3rc4/bin/
# mpirun -np 6 \
#        -H $ip_gx8:6 \
#        -bind-to none -map-by slot \
#        -mca pml ob1 -mca btl ^openib \
#        -mca btl_tcp_if_exclude docker0,lo\
#        -x NCCL_SOCKET_IFNAME=enp10s0f0 \
#        CUDA_VISIBLE_DEVICES=0,2,3,5,6,7 \
#        $conda_env_gx8 $exe_dir_gx8 --log_dir $log_dir
