from os import mkdir
from os.path import join
from shutil import copyfile
from shutil import rmtree

raw_home = '/home/tan/tony/kitti_raw'
project_home = '/home/tan/tony/dv-det'
task = 'validation'

labels_input_dir = join(raw_home, 'training', 'label_2')
labels_output_dir = join(project_home, 'eval/kitti/data/label_2')

selected_frame_txt = '/home/tan/tony/dv-det/data/data_split_eval/validation.txt'

try: rmtree(labels_output_dir)
except: pass
try: mkdir(labels_output_dir)
except: pass


if __name__ == '__main__':
    with open(selected_frame_txt, 'r') as f:
        for i, frame_id in enumerate(f.readlines()):
            copyfile(src=join(labels_input_dir, '%06d.txt'%int(frame_id)),
                     dst=join(labels_output_dir, '%06d.txt'%i))


