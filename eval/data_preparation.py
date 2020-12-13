import numpy as np
from os import mkdir
from os.path import join
from tqdm import tqdm
import logging
from data.kitti_generator import Dataset

task = 'validation'
validation = True
evaluation = False
output_home = '/home/tan/tony/dv-det/eval/data'


DatasetEval = Dataset(task=task,
                      validation=validation,
                      evaluation=evaluation)

try:
    mkdir(output_home)
except:
    logging.warning('Directory: {} already exists.'.format(output_home))

if __name__ == '__main__':

    output_coors, output_features, output_num_list, output_bboxes = [], [], [], []
    for _ in tqdm(range(DatasetEval.total_data_length)):
        if not evaluation:
            coors, features, num_list, bboxes = next(DatasetEval.valid_generator())
        else:
            coors, features, num_list = next(DatasetEval.valid_generator())
        output_coors.append(coors)
        output_features.append(features)
        output_num_list.append(num_list)
        if not evaluation:
            output_bboxes.append(bboxes)

    np.save(join(output_home, 'output_coors.npy'), np.array(output_coors, dtype=object))
    np.save(join(output_home, 'output_features.npy'), np.array(output_features, dtype=object))
    np.save(join(output_home, 'output_num_list.npy'), np.array(output_num_list, dtype=object))
    if evaluation:
        np.save(join(output_home, 'output_bboxes.npy'), np.array(output_bboxes, dtype=object))