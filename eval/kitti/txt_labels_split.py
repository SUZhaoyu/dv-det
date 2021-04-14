from shutil import copyfile
from os import mkdir
from os.path import join
from shutil import rmtree
from tqdm import tqdm
import logging

raw_home = '/media/data1/kitti_raw'
project_home = '/home/tan/tony/dv-det'
task = 'validation'

labels_input_dir = join(raw_home, 'training', 'label_2')
labels_output_dir = join(project_home, 'eval', 'kitti', 'data', 'label_2')
label_file_list_txt = join(project_home, 'data', 'data_split_half', task + '.txt')

try: rmtree(labels_output_dir)
except: pass

try: mkdir(labels_output_dir)
except: logging.warning('Directory: {} already exists.'.format(labels_output_dir))

file_id = 0
if __name__ == '__main__':
    with open(label_file_list_txt, 'r') as f:
        for frame_id in tqdm(f.readlines()):
            copyfile(src=join(labels_input_dir, '{}.txt'.format(frame_id[:6])),
                     dst=join(labels_output_dir, '%06d.txt'%(int(file_id))))
            file_id += 1

