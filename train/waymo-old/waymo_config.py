import os

import numpy as np

batch_size = 2
bbox_padding = 256
aug_config = {'nbbox': bbox_padding,
              # 'train': 798 * 200 * 0.2 / 8 / batch_size,
              # 'val': 202 * 200 * 0.2 / 8 / batch_size,
              'rotate_range': np.pi * 2,
              'rotate_mode': 'u',
              'scale_range': 0.05,
              'scale_mode': 'u',
              'drop_out': 0.1,
              'flip': False,
              'shuffle': True,
              'paste_augmentation': True,
              'paste_instance_num': 128,
              'maximum_interior_points': 40,
              'normalization': 'channel_std'}

dimension_training = [180., 180., 8.]
offset_training = [90., 90., 3.5]

# dimension_training = [152., 152., 6.5]
# offset_training = [76., 76., 2.5]


anchor_size = [2.1, 4.8, 1.75]
grid_buffer_size = 3
output_pooling_size = 5


local = False

gpu_list = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
            8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6, 15: 7}

model_file_name = os.path.basename(__file__).split('.')[0] + '.py'
model_file_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(model_file_dir, model_file_name)


decay_epochs = 5
init_lr = 1e-4
lr_decay = 0.5
lr_scale = True
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
normalization = "channel_std"
num_worker = 5
weighted = False
use_l2 = True
output_attr = 8
# stage1_training_epoch = 25
total_epoch = 300

roi_thres = 0.5
iou_thres = 0.55
max_length = 512
roi_voxel_size = 5


base_params_inference = {'base_00': {'subsample_res': 0.10, 'c_out':  16, 'kernel_res': 0.10, 'concat': True},
                         'base_01': {'subsample_res': None, 'c_out':  16, 'kernel_res': 0.20, 'concat': False},
                         'base_02': {'subsample_res': None, 'c_out':  16, 'kernel_res': 0.20, 'concat': False},
                         'base_03': {'subsample_res': 0.20, 'c_out':  16, 'kernel_res': 0.20, 'concat': True},
                         'base_04': {'subsample_res': None, 'c_out':  32, 'kernel_res': 0.40, 'concat': False},
                         'base_05': {'subsample_res': None, 'c_out':  32, 'kernel_res': 0.40, 'concat': False},
                         'base_06': {'subsample_res': 0.40, 'c_out':  32, 'kernel_res': 0.40, 'concat': True},
                         'base_07': {'subsample_res': None, 'c_out':  64, 'kernel_res': [0.80, 0.80, 0.40], 'concat': False},
                         'base_08': {'subsample_res': None, 'c_out':  64, 'kernel_res': [0.80, 0.80, 0.40], 'concat': False},
                         'base_09': {'subsample_res': 0.60, 'c_out':  64, 'kernel_res': [0.80, 0.80, 0.40], 'concat': True},
                         'base_10': {'subsample_res': None, 'c_out': 128, 'kernel_res': [1.60, 1.60, 0.80], 'concat': False},
                         'base_11': {'subsample_res': None, 'c_out': 128, 'kernel_res': [1.60, 1.60, 0.80], 'concat': False},
                         'base_12': {'subsample_res': None, 'c_out': 128, 'kernel_res': [1.60, 1.60, 0.80], 'concat': True}}


stage2_input_channels = 2
for layer_name in base_params_inference.keys():
    stage2_input_channels += base_params_inference[layer_name]['c_out']


refine_params = {'c_out': 256, 'kernel_size': 3, 'padding': 0.}
