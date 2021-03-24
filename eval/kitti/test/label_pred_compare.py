import os
from os import listdir
from os.path import join

import numpy as np
from point_viz.converter import PointvizConverter
from tqdm import tqdm

from data.utils.normalization import convert_threejs_bbox_with_colors, convert_threejs_coors

os.system("rm -r {}".format('/home/tan/tony/threejs/kitti-compare'))
Converter = PointvizConverter(home='/home/tan/tony/threejs/kitti-compare')

TASK = 'validation'
label_home = '/home/tan/tony/dv-det/eval/label_2'
pred_home = '/home/tan/tony/dv-det/eval/txt'
npy_home = '/home/tan/tony/dv-det/dataset'

label_file_list = listdir(label_home)
trans_matrix = np.load(join(npy_home, 'trans_matrix_{}.npy'.format(TASK)), allow_pickle=True)
point_cloud = np.load(join(npy_home, 'lidar_points_{}.npy'.format(TASK)), allow_pickle=True)

assert len(label_file_list) == len(point_cloud)

def load_bbox(dir, trans_list):
    output_bbox = []
    P2, R0_rect, Tr_velo_to_cam = trans_list
    inv_trans_matrix = np.linalg.inv(R0_rect.dot(Tr_velo_to_cam))
    with open(dir, 'r') as f:
        for text_line in f.readlines():
            line = text_line.split(' ')
            x, y, z, _ = inv_trans_matrix.dot(np.array([float(line[11]),
                                                        float(line[12]),
                                                        float(line[13]),
                                                        1.])).transpose().tolist()  # from camera coors to LiDAR coors

            h = float(line[8])
            w = float(line[9])
            l = float(line[10])
            r = -float(line[14])
            z += h / 2.
            output_bbox.append([w, l, h, x, y, z, r, 0, 0])

    return output_bbox


if __name__ == '__main__':
    for i, name in enumerate(tqdm(label_file_list)):
        label_txt = join(label_home, name)
        pred_txt = join(pred_home, name)

        label_bbox = load_bbox(label_txt, trans_matrix[i])
        pred_bbox = load_bbox(pred_txt, trans_matrix[i])

        label_bbox_params = convert_threejs_bbox_with_colors(label_bbox, color='blue') if len(label_bbox) > 0 else []
        pred_bbox_params = convert_threejs_bbox_with_colors(pred_bbox, color='red') if len(pred_bbox) > 0 else []
        points = point_cloud[i]

        Converter.compile(task_name="ID_%06d_%03d" % (i, len(pred_bbox)),
                          coors=convert_threejs_coors(points[:, :3]),
                          bbox_params=pred_bbox_params + label_bbox_params)
