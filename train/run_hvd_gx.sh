#!/usr/bin/env bash
#exe_file="train_stage1_anchor_x2.py"
exe_file="kitti/train_stage1.py"
#exe_file="train_stage1.py"
pkill -f -9 $exe_file

HOME="dv-det"
checkpoints="ckpt-kitti"

root_gx4="/home/tan/tony"
root_gx6="/home/tan/tony"
ip_gx4="192.168.66.54"
ip_gx6="192.168.66.56"

# rsync -avz -W -e ssh --progress \
#                      --exclude='*.pyc' \
#                      --exclude='checkpoints' \
#                      --exclude='build' \
#                      --exclude='ckpt_archive' \
#                      --exclude='eval' \
#                      --exclude='ckpt_arxiv' \
#                      --exclude='img_*.npy' \
#                      $root_gx4/$HOME tan@$ip_gx6:$root_gx6

echo "Pushing Completed!"


conda_env_gx4="/home/tan/anaconda3/envs/detection/bin/python"
home_dir_gx4="$root_gx4/$HOME"
exe_dir_gx4="$home_dir_gx4/train/$exe_file"

conda_env_gx6="/home/tan/anaconda3/envs/detection/bin/python"
home_dir_gx6="$root_gx6/$HOME"
exe_dir_gx6="$home_dir_gx6/train/$exe_file"

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

# mpirun -np 8 \
#        -H $ip_gx4:8\
#        -bind-to none -map-by slot \
#        -mca pml ob1 -mca btl ^openib \
#        -mca btl_tcp_if_exclude docker0,lo,enp1s0f0,enp1s0f1\
#        -x NCCL_SOCKET_IFNAME=enp130s0 \
#        $conda_env_gx4 $exe_dir_gx4 --log_dir $log_dir :\
#        -np 8 \
#        -H $ip_gx6:8\
#        -bind-to none -map-by slot \
#        -mca pml ob1 -mca btl ^openib \
#        -mca btl_tcp_if_exclude docker0,lo,enp10s0f0\
#        -x NCCL_SOCKET_IFNAME=enp130s0 \
#        $conda_env_gx6 $exe_dir_gx6


#horovodrun -np 16 -H $ip_gx4:8,$ip_gx6:8 $conda_env_gx4 $exe_dir_gx4 --log_dir $log_dir
horovodrun -np 8 -H $ip_gx4:8 $conda_env_gx4 $exe_dir_gx4 --log_dir $log_dir

# /usr/mpi/gcc/openmpi-4.0.3rc4/bin/
# mpirun -np 6 \
#        -H $ip_gx6:6 \
#        -bind-to none -map-by slot \
#        -mca pml ob1 -mca btl ^openib \
#        -mca btl_tcp_if_exclude docker0,lo\
#        -x NCCL_SOCKET_IFNAME=enp10s0f0 \
#        CUDA_VISIBLE_DEVICES=0,2,3,5,6,7 \
#        $conda_env_gx6 $exe_dir_gx6 --log_dir $log_dir
