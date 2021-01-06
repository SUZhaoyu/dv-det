import logging
from os import mkdir
from os.path import join

import numpy as np
from tqdm import tqdm

from data.kitti_generator import Dataset

task = 'validation'
validation = True
evaluation = False
output_home = '/home/tan/tony/dv-det/eval/data'


DatasetEval = Dataset(task=task,
                      batch_size=1,
                      validation=validation,
                      evaluation=evaluation)

try:
    mkdir(output_home)
except:
    logging.warning('Directory: {} already exists.'.format(output_home))

if __name__ == '__main__':

    input_coors, input_features, input_num_list, input_bboxes = [], [], [], []
    for _ in tqdm(range(DatasetEval.batch_sum)):
        if not evaluation:
            coors, features, num_list, bboxes = next(DatasetEval.valid_generator())
        else:
            coors, features, num_list = next(DatasetEval.valid_generator())
        input_coors.append(coors)
        input_features.append(features)
        input_num_list.append(num_list)
        if not evaluation:
            input_bboxes.append(bboxes)

    np.save(join(output_home, 'input_coors.npy'), np.array(input_coors, dtype=object))
    np.save(join(output_home, 'input_features.npy'), np.array(input_features, dtype=object))
    np.save(join(output_home, 'input_num_list.npy'), np.array(input_num_list, dtype=object))
    if evaluation:
        np.save(join(output_home, 'input_bboxes.npy'), np.array(input_bboxes, dtype=object))