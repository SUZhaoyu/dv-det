from os import mkdir
from os.path import join
from shutil import copyfile
from shutil import rmtree

raw_home = '/home/tan/tony/kitti_raw'
project_home = '/home/tan/tony/dv-det'
task = 'validation'

labels_input_dir = join(project_home, 'eval', 'label_2')
predictions_input_dir = join(project_home, 'eval', 'txt')
labels_output_dir = join(project_home, 'eval', 'label_selected')
predictions_output_dir = join(project_home, 'eval', 'txt_selected')

selected_frame_txt = join(project_home, 'eval', 'test/selected_frame.txt')

try: rmtree(labels_output_dir)
except: pass
try: mkdir(labels_output_dir)
except: pass

try: rmtree(predictions_output_dir)
except: pass
try: mkdir(predictions_output_dir)
except: pass

if __name__ == '__main__':
    with open(selected_frame_txt, 'r') as f:
        for i, frame_id in enumerate(f.readlines()):
            copyfile(src=join(labels_input_dir, '%06d.txt'%int(frame_id)),
                     dst=join(labels_output_dir, '%06d.txt'%i))
            copyfile(src=join(predictions_input_dir, '%06d.txt' % int(frame_id)),
                     dst=join(predictions_output_dir, '%06d.txt' % i))

