from os.path import join

import numpy as np

dataset_home = '/media/data1/processed/training'

data = np.load(join(dataset_home, 'train_lidars_643.npy'), allow_pickle=True)
print(" ")