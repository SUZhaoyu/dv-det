import os

import numpy as np

aug_config = {'nbbox': 128,
              'rotate_range': np.pi / 4,
              # 'rotate_range': 0,
              'rotate_mode': 'u',
              'scale_range': 0.05,
              'scale_mode': 'u',
              'drop_out': 0.1,
              'flip': False,
              'shuffle': True,
              'paste_augmentation': True,
              'paste_instance_num': 64,
              'maximum_interior_points': 100,
              'normalization': None}

dimension_training = [140., 140., 9.]
offset_training = [50., 50., 5.]

# dimension_training = [100., 100., 9.]
# offset_training = [10., 10., 5.]

# dimension_training = [72, 80.0, 9.0]
# offset_training = [0., 40.0, 5.0]


anchor_size = [1.6, 3.9, 1.5]
grid_buffer_size = 3
output_pooling_size = 5

diff_thres = 3
cls_thres = 0

local = False

gpu_list = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
            8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6, 15: 7}

model_file_name = os.path.basename(__file__).split('.')[0] + '.py'
model_file_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(model_file_dir, model_file_name)

bbox_padding = aug_config['nbbox']
batch_size = 8
decay_epochs = 10

init_lr_stage1 = 1e-4
lr_scale_stage1 = True

init_lr_stage2 = 2e-4
lr_scale_stage2 = False

lr_decay = 0.5
lr_warm_up = False
cls_loss_scale = 1.
weight_decay = 5e-4
valid_interval = 5
use_trimmed_foreground = False
paste_augmentation = False
use_la_pooling = False
norm_angle = False
xavier = True
stddev = 1e-3
activation = 'relu'
normalization = None
num_worker = 5
weighted = False
use_l2 = True
output_attr = 8
# stage1_training_epoch = 25
total_epoch = 300

roi_thres = 0.5
iou_thres = 0.5
max_length = 256
roi_voxel_size = 7

# base_params_training = {'base_0': {'subsample_res': 0.10, 'c_out':  16, 'kernel_res': 0.10, 'padding': -1.},
#                         'base_1': {'subsample_res': 0.10, 'c_out':  32, 'kernel_res': 0.20, 'padding':  0.},
#                         'base_2': {'subsample_res': 0.20, 'c_out':  32, 'kernel_res': 0.20, 'padding':  0.},
#                         'base_3': {'subsample_res': 0.20, 'c_out':  64, 'kernel_res': 0.40, 'padding':  0.},
#                         'base_4': {'subsample_res': 0.40, 'c_out':  64, 'kernel_res': 0.40, 'padding':  0.},
#                         'base_5': {'subsample_res': 0.40, 'c_out': 128, 'kernel_res': 0.60, 'padding':  0.},
#                         'base_6': {'subsample_res': 0.60, 'c_out': 128, 'kernel_res': 0.60, 'padding':  0.},
#                         'base_7': {'subsample_res': 0.60, 'c_out': 256, 'kernel_res': 0.80, 'padding':  0.}}
# rpn_params_training = {'subsample_res': 0.80, 'c_out': 256, 'kernel_res': 0.80, 'padding': 0.}

base_params_inference = {'base_0': {'subsample_res': 0.10, 'c_out':  16, 'kernel_res': 0.10, 'padding': 0.},
                         'base_1': {'subsample_res': 0.10, 'c_out':  32, 'kernel_res': 0.20, 'padding': 0.},
                         'base_2': {'subsample_res': 0.20, 'c_out':  32, 'kernel_res': 0.20, 'padding': 0.},
                         'base_3': {'subsample_res': 0.20, 'c_out':  64, 'kernel_res': 0.40, 'padding': 0.},
                         'base_4': {'subsample_res': 0.40, 'c_out':  64, 'kernel_res': 0.40, 'padding': 0.},
                         'base_5': {'subsample_res': 0.40, 'c_out': 128, 'kernel_res': 0.80, 'padding': 0.},
                         'base_6': {'subsample_res': 0.60, 'c_out': 128, 'kernel_res': 0.80, 'padding': 0.},
                         'base_7': {'subsample_res': 0.60, 'c_out': 256, 'kernel_res': 1.20, 'padding': 0.}}
rpn_params_inference = {'subsample_res': 0.80, 'c_out': 256, 'kernel_res': 1.20, 'padding': 0.}

# base_params_inference = {'base_0': {'subsample_res': 0.10, 'c_out':  16, 'kernel_res': 0.10, 'padding': -1.},
#                          'base_1': {'subsample_res': None, 'c_out':  32, 'kernel_res': 0.20, 'padding':  0.},
#                          'base_2': {'subsample_res': 0.20, 'c_out':  32, 'kernel_res': None, 'padding':  0.},
#                          'base_3': {'subsample_res': None, 'c_out':  64, 'kernel_res': 0.40, 'padding':  0.},
#                          'base_4': {'subsample_res': 0.40, 'c_out':  64, 'kernel_res': None, 'padding':  0.},
#                          'base_5': {'subsample_res': None, 'c_out': 128, 'kernel_res': 0.80, 'padding':  0.},
#                          'base_6': {'subsample_res': 0.60, 'c_out': 128, 'kernel_res': None, 'padding':  0.},
#                          'base_7': {'subsample_res': None, 'c_out': 256, 'kernel_res': 1.20, 'padding':  0.}}
# rpn_params_inference = {'subsample_res': 0.80, 'c_out': 256, 'kernel_res': None, 'padding': 0.}

# base_params_inference = {'base_0': {'subsample_res': 0.10, 'c_out':  16, 'kernel_res': 0.10, 'padding':  0.},
#                          'base_1': {'subsample_res': 0.20, 'c_out':  32, 'kernel_res': 0.20, 'padding':  0.},
#                          'base_2': {'subsample_res': 0.40, 'c_out':  64, 'kernel_res': 0.40, 'padding':  0.},
#                          'base_3': {'subsample_res': 0.60, 'c_out': 128, 'kernel_res': 0.60, 'padding':  0.},
#                          'base_4': {'subsample_res': 0.60, 'c_out': 256, 'kernel_res': 0.80, 'padding':  0.}}
# rpn_params_inference = {'subsample_res': 0.80, 'c_out': 256, 'kernel_res': 0.80, 'padding': 0.}

# base_params_inference = {'base_0': {'subsample_res': 0.10, 'c_out':  16, 'kernel_res': 0.10, 'padding':  0.},
#                          'base_1': {'subsample_res': 0.20, 'c_out':  32, 'kernel_res': 0.20, 'padding':  0.},
#                          'base_2': {'subsample_res': 0.40, 'c_out':  64, 'kernel_res': 0.40, 'padding':  0.},
#                          'base_3': {'subsample_res': 0.60, 'c_out': 128, 'kernel_res': 0.60, 'padding':  0.},
#                          'base_4': {'subsample_res': 0.80, 'c_out': 256, 'kernel_res': 0.80, 'padding':  0.}}
# rpn_params_inference = {'subsample_res': 1.20, 'c_out': 256, 'kernel_res': 1.20, 'padding': 0.}

refine_params = {'c_out': 256, 'kernel_size': 3, 'padding': 0.}

# grid_dimensions = np.array(np.array(dimension_training) / base_params_inference['base_0']['kernel_res'], dtype=np.int32)
# maximum_grid_num = dimension_training[0] * dimension_training[1] * dimension_training[2] * batch_size
# if maximum_grid_num > 1 << 31 - 1:
#     raise ValueError("Grid number exceed INT32 range: {} x {} x {} x {}",
#                      batch_size, dimension_training[0], dimension_training[1], dimension_training[2])
