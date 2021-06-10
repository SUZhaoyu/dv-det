import os

import math

aug_config = {'nbbox': 256,
              'rotate_range': math.pi / 4,
              # 'rotate_range': 0,
              'rotate_mode': 'u',
              'scale_range': 0.05,
              'scale_mode': 'u',
              'drop_out': 0.1,
              'flip': False,
              'shuffle': True,
              'paste_augmentation': True,
              'paste_instance_num': 128,
              'maximum_interior_points': 100,
              'normalization': None}

# dimension_training = [140., 140., 4.]
# offset_training = [50., 50., 3.]

# dimension_training = [100., 100., 9.]
# offset_training = [10., 10., 5.]

dimension_training = [74, 84.0, 4.]
offset_training = [2., 42.0, 3.]

anchor_size = [1.6, 3.9, 1.5]
anchor_params = [[1.6, 3.9, 1.5, -1., 0.],
                 [1.6, 3.9, 1.5, -1., math.pi/2.]]
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
decay_epochs = 5

batch_size_stage1 = 4
init_lr_stage1 = 2e-3
lr_scale_stage1 = True

batch_size_stage2 = 4
init_lr_stage2 = 2e-4
lr_scale_stage2 = False

lr_decay = 0.5
lr_warm_up = True
cls_loss_scale = 1.
weight_decay = 5e-4
valid_interval = 5
use_trimmed_foreground = False
paste_augmentation = False
use_la_pooling = False
norm_angle = False
xavier = False
stddev = 1e-3
activation = 'relu'
normalization = None
num_worker = 5
weighted = False
use_l2 = True
output_attr = 8
# stage1_training_epoch = 25
total_epoch = 300

roi_thres = 0.3
iou_thres = 0.55
max_length = 256
roi_voxel_size = 5

# base_params_inference = {'base_00': {'subsample_res': 0.05, 'c_out':  16, 'kernel_res': 0.05, 'concat': False},
#                          'base_02': {'subsample_res': None, 'c_out':  16, 'kernel_res': 0.10, 'concat': False},
#                          'base_03': {'subsample_res': 0.10, 'c_out':  16, 'kernel_res': 0.10, 'concat': False},
#                          'base_05': {'subsample_res': None, 'c_out':  32, 'kernel_res': 0.20, 'concat': False},
#                          'base_06': {'subsample_res': 0.20, 'c_out':  32, 'kernel_res': 0.20, 'concat': True},
#                          'base_07': {'subsample_res': None, 'c_out':  64, 'kernel_res': 0.40, 'concat': False},
#                          'base_08': {'subsample_res': 0.40, 'c_out':  64, 'kernel_res': 0.40, 'concat': True},
#                          'base_09': {'subsample_res': None, 'c_out':  64, 'kernel_res': [0.80, 0.80, 0.40], 'concat': False},
#                          'base_10': {'subsample_res': 0.60, 'c_out':  64, 'kernel_res': [0.80, 0.80, 0.40], 'concat': True},
#                          'base_11': {'subsample_res': None, 'c_out': 128, 'kernel_res': [1.60, 1.60, 0.80], 'concat': False},
#                          'base_13': {'subsample_res': None, 'c_out': 128, 'kernel_res': [1.60, 1.60, 0.80], 'concat': True}}

# base_params_inference = {'base_00': {'subsample_res': 0.10, 'c_out':  16, 'kernel_res': 0.10, 'concat': True},
#                          'base_02': {'subsample_res': None, 'c_out':  16, 'kernel_res': 0.20, 'concat': False},
#                          'base_03': {'subsample_res': None, 'c_out':  16, 'kernel_res': 0.20, 'concat': False},
#                          'base_04': {'subsample_res': 0.20, 'c_out':  16, 'kernel_res': 0.20, 'concat': True},
#                          'base_05': {'subsample_res': None, 'c_out':  32, 'kernel_res': 0.40, 'concat': False},
#                          'base_06': {'subsample_res': None, 'c_out':  32, 'kernel_res': 0.40, 'concat': False},
#                          'base_07': {'subsample_res': 0.40, 'c_out':  32, 'kernel_res': 0.40, 'concat': True},
#                          'base_08': {'subsample_res': None, 'c_out':  64, 'kernel_res': [0.80, 0.80, 0.40], 'concat': False},
#                          'base_09': {'subsample_res': None, 'c_out':  64, 'kernel_res': [0.80, 0.80, 0.40], 'concat': False},
#                          'base_10': {'subsample_res': 0.60, 'c_out':  64, 'kernel_res': [0.80, 0.80, 0.40], 'concat': True},
#                          'base_11': {'subsample_res': None, 'c_out': 128, 'kernel_res': [1.60, 1.60, 0.80], 'concat': False},
#                          'base_12': {'subsample_res': None, 'c_out': 128, 'kernel_res': [1.60, 1.60, 0.80], 'concat': False},
#                          'base_13': {'subsample_res': None, 'c_out': 128, 'kernel_res': [1.60, 1.60, 0.80], 'concat': True}}





base_params_inference = {'base_00': {'subsample_res': 0.10, 'c_out':  16, 'kernel_res': 0.10, 'concat': True},
                         'base_02': {'subsample_res': None, 'c_out':  16, 'kernel_res': 0.20, 'concat': False},
                         'base_03': {'subsample_res': None, 'c_out':  16, 'kernel_res': None, 'concat': False},
                         'base_04': {'subsample_res': 0.20, 'c_out':  16, 'kernel_res': None, 'concat': True},
                         'base_05': {'subsample_res': None, 'c_out':  32, 'kernel_res': 0.40, 'concat': False},
                         'base_06': {'subsample_res': None, 'c_out':  32, 'kernel_res': None, 'concat': False},
                         'base_07': {'subsample_res': 0.40, 'c_out':  32, 'kernel_res': None, 'concat': True},
                         'base_08': {'subsample_res': None, 'c_out':  64, 'kernel_res': [0.80, 0.80, 0.40], 'concat': False},
                         'base_09': {'subsample_res': None, 'c_out':  64, 'kernel_res': None, 'concat': False},
                         'base_10': {'subsample_res': 0.60, 'c_out':  64, 'kernel_res': None, 'concat': True}}
                         # 'base_11': {'subsample_res': None, 'c_out': 128, 'kernel_res': [1.60, 1.60, 0.80], 'concat': False},
                         # 'base_12': {'subsample_res': None, 'c_out': 128, 'kernel_res': None, 'concat': False},
                         # 'base_13': {'subsample_res': None, 'c_out': 128, 'kernel_res': None, 'concat': True}}



refine_params = {'c_out': 128, 'kernel_size': 3, 'padding': 0.}

# grid_dimensions = np.array(np.array(dimension_training) / base_params_inference['base_0']['kernel_res'], dtype=np.int32)
# maximum_grid_num = dimension_training[0] * dimension_training[1] * dimension_training[2] * batch_size
# if maximum_grid_num > 1 << 31 - 1:
#     raise ValueError("Grid number exceed INT32 range: {} x {} x {} x {}",
#                      batch_size, dimension_training[0], dimension_training[1], dimension_training[2])
