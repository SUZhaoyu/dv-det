from os import mkdir
import os
from os.path import join
from shutil import copyfile
from shutil import rmtree
from os import listdir
from tqdm import tqdm

project_home = '/home/tan/tony/dv-det'
task = 'validation'

pred_input_dir = join(project_home, 'eval/kitti/data/pv_rcnn-val')
pred_output_dir = join(project_home, 'eval/kitti/data/compare_txt')

try: rmtree(pred_output_dir)
except: pass
try: mkdir(pred_output_dir)
except: pass


if __name__ == '__main__':
    file_list = listdir(pred_input_dir)
    for i in tqdm(range(len(file_list))):
        os.system('cp {} {}'.format(join(pred_input_dir, file_list[i]), join(pred_output_dir, '%06d.txt' % i)))

    # with open(selected_frame_txt, 'r') as f:
    #     for i, frame_id in enumerate(f.readlines()):
    #         copyfile(src=join(labels_input_dir, '%06d.txt' % int(frame_id)),
    #                  dst=join(labels_output_dir, '%06d.txt' % i))

