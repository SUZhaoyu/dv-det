import os

import numpy as np

batch_size = 8
bbox_padding = 256
aug_config = {'nbbox': bbox_padding,
              'train': 798 * 200 * 0.2 / 8 / batch_size,
              'val': 202 * 200 * 0.2 / 8 / batch_size,
              'rotate_range': np.pi * 2,
              'rotate_mode': 'u',
              'scale_range': 0.05,
              'scale_mode': 'u',
              'drop_out': 0.1,
              'flip': False,
              'shuffle': True,
              'paste_augmentation': True,
              'paste_instance_num': 16,
              'maximum_interior_points': 40,
              'normalization': 'channel_std'}

dimension_training = [180., 180., 8.]
offset_training = [90., 90., 3.0]

# dimension_inference = [70.4, 80.0, 4.0]
# offset_inference = [0., 40.0, 3.0]

anchor_size = [2.1, 4.8, 1.75]


local = False

gpu_list = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
            8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6, 15: 7}

model_file_name = os.path.basename(__file__).split('.')[0] + '.py'
model_file_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(model_file_dir, model_file_name)


decay_epochs = 5
init_lr = 2e-4
lr_decay = 0.5
lr_scale = False
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

roi_thres = 0.4
iou_thres = 0.6
max_length = 768
roi_voxel_size = 5

# base_params_training = {'base_0': {'subsample_res': 0.10, 'c_out':  16, 'kernel_res': 0.10, 'padding': -1.},
#                         'base_1': {'subsample_res': 0.10, 'c_out':  32, 'kernel_res': 0.20, 'padding':  0.},
#                         'base_2': {'subsample_res': 0.20, 'c_out':  32, 'kernel_res': 0.20, 'padding':  0.},
#                         'base_3': {'subsample_res': 0.20, 'c_out':  64, 'kernel_res': 0.40, 'padding':  0.},
#                         'base_4': {'subsample_res': 0.40, 'c_out':  64, 'kernel_res': 0.40, 'padding':  0.},
#                         'base_5': {'subsample_res': 0.40, 'c_out': 128, 'kernel_res': 0.60, 'padding':  0.},
#                         'base_6': {'subsample_res': 0.60, 'c_out': 128, 'kernel_res': 0.60, 'padding':  0.},
#                         'base_7': {'subsample_res': 0.60, 'c_out': 256, 'kernel_res': 0.80, 'padding':  0.}}
# rpn_params_training = {'subsample_res': 0.80, 'c_out': 256, 'kernel_res': 0.80, 'padding': 0.}

# base_params_inference = {'base_0': {'subsample_res': 0.10, 'c_out':  16, 'kernel_res': 0.10, 'padding': -1.},
#                         'base_1': {'subsample_res': 0.10, 'c_out':  32, 'kernel_res': 0.20, 'padding':  0.},
#                         'base_2': {'subsample_res': 0.20, 'c_out':  32, 'kernel_res': 0.20, 'padding':  0.},
#                         'base_3': {'subsample_res': 0.20, 'c_out':  64, 'kernel_res': 0.40, 'padding':  0.},
#                         'base_4': {'subsample_res': 0.40, 'c_out':  64, 'kernel_res': 0.40, 'padding':  0.},
#                         'base_5': {'subsample_res': 0.40, 'c_out': 128, 'kernel_res': 0.60, 'padding':  0.},
#                         'base_6': {'subsample_res': 0.60, 'c_out': 128, 'kernel_res': 0.60, 'padding':  0.},
#                         'base_7': {'subsample_res': 0.60, 'c_out': 256, 'kernel_res': 0.80, 'padding':  0.}}
# rpn_params_inference = {'subsample_res': 0.80, 'c_out': 256, 'kernel_res': 0.80, 'padding': 0.}

# base_params_inference = {'base_0': {'subsample_res': 0.10, 'c_out':  16, 'kernel_res': 0.10, 'padding':  0.},
#                          'base_1': {'subsample_res': None, 'c_out':  32, 'kernel_res': 0.20, 'padding':  0.},
#                          'base_2': {'subsample_res': 0.20, 'c_out':  32, 'kernel_res': 0.20, 'padding':  0.},
#                          'base_3': {'subsample_res': None, 'c_out':  64, 'kernel_res': 0.40, 'padding':  0.},
#                          'base_4': {'subsample_res': 0.40, 'c_out':  64, 'kernel_res': 0.40, 'padding':  0.},
#                          'base_5': {'subsample_res': None, 'c_out': 128, 'kernel_res': 0.60, 'padding':  0.},
#                          'base_6': {'subsample_res': 0.60, 'c_out': 128, 'kernel_res': 0.60, 'padding':  0.},
#                          'base_7': {'subsample_res': None, 'c_out': 256, 'kernel_res': 0.80, 'padding':  0.}}
# rpn_params_inference = {'subsample_res': 0.80, 'c_out': 256, 'kernel_res': 0.80, 'padding': 0.}

base_params_inference = {'base_0': {'subsample_res': 0.10, 'c_out':  16, 'kernel_res': 0.10, 'padding':  0.},
                         'base_1': {'subsample_res': 0.20, 'c_out':  32, 'kernel_res': 0.20, 'padding':  0.},
                         'base_2': {'subsample_res': 0.40, 'c_out':  64, 'kernel_res': 0.40, 'padding':  0.},
                         'base_3': {'subsample_res': 0.60, 'c_out': 128, 'kernel_res': 0.60, 'padding':  0.},
                         'base_4': {'subsample_res': 0.80, 'c_out': 256, 'kernel_res': 0.80, 'padding':  0.}}
rpn_params_inference = {'subsample_res': 1.20, 'c_out': 256, 'kernel_res': 1.20, 'padding': 0.}

# base_params_inference = {'base_0': {'subsample_res': 0.05, 'c_out':  16, 'kernel_res': 0.05, 'padding': -1.},
#                          'base_1': {'subsample_res': 0.10, 'c_out':  16, 'kernel_res': 0.10, 'padding':  0.},
#                          'base_2': {'subsample_res': 0.20, 'c_out':  32, 'kernel_res': 0.20, 'padding':  0.},
#                          'base_3': {'subsample_res': 0.40, 'c_out':  64, 'kernel_res': 0.40, 'padding':  0.},
#                          'base_4': {'subsample_res': 0.60, 'c_out': 128, 'kernel_res': 0.60, 'padding':  0.},
#                          'base_5': {'subsample_res': 0.80, 'c_out': 256, 'kernel_res': 0.80, 'padding':  0.},}
# rpn_params_inference = {'subsample_res': 0.80, 'c_out': 256, 'kernel_res': 0.80, 'padding': 0.}

# base_params_inference = {'base_0': {'subsample_res': 0.10, 'c_out':  16, 'kernel_res': 0.10, 'padding': -1.},
#                          'base_1': {'subsample_res': 0.20, 'c_out':  32, 'kernel_res': 0.20, 'padding':  0.},
#                          'base_2': {'subsample_res': 0.40, 'c_out':  64, 'kernel_res': 0.40, 'padding':  0.},
#                          'base_3': {'subsample_res': 0.60, 'c_out': 128, 'kernel_res': 0.60, 'padding':  0.}}
# rpn_params_inference = {'subsample_res': 0.80, 'c_out': 128, 'kernel_res': 0.80, 'padding': 0.}

refine_params = {'c_out': 256, 'kernel_size': 3, 'padding': 0.}
