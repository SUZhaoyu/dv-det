import os

import numpy as np

aug_config = {'nbbox': 64,
              'rotate_range': np.pi / 4,
              'rotate_mode': 'u',
              'scale_range': 0.05,
              'scale_mode': 'u',
              'drop_out': 0.1,
              'flip': False,
              'shuffle': True,
              'paste_augmentation': True,
              'paste_instance_num': 32,
              'maximum_interior_points': 40,
              'normalization': None}

dimension_training = [100, 160.0, 9.0]
offset_training = [10., 60.0, 5.0]

dimension_testing = [70.4, 80.0, 4.0]
offset_testing = [0., 40.0, 3.0]


local = False

gpu_list = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
            8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6, 15: 7}

model_file_name = os.path.basename(__file__).split('.')[0] + '.py'
model_file_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(model_file_dir, model_file_name)

bbox_padding = 64
batch_size = 8
decay_epochs = 20
init_lr = 1e-4
lr_decay = 0.5
lr_scale = True
lr_warm_up = False
cls_loss_scale = 1.
weight_decay = 1e-3
valid_interval = 5
use_trimmed_foreground = False
paste_augmentation = True
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
stage1_training_epoch = 0
total_epoch = 500


roi_thres = 0.3
roi_voxel_size = 5
base_params = {'base_0': {'subsample_res': 0.1, 'c_out': 16, 'kernel_res': 0.1, 'padding': -1.},
               'base_1': {'subsample_res': 0.2, 'c_out': 32, 'kernel_res': 0.2, 'padding': 0.},
               'base_2': {'subsample_res': 0.4, 'c_out': 64, 'kernel_res': 0.4, 'padding': 0.},
               'base_3': {'subsample_res': 0.6, 'c_out': 128, 'kernel_res': 0.6, 'padding': 0.}}

rpn_params = {'subsample_res': 0.8, 'c_out': 128, 'kernel_res': 0.8, 'padding': 0.}
refine_params = {'c_out': 256, 'kernel_size': 3, 'padding': 0.}
