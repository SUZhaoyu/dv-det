import numpy as np
import cv2
import os
from os.path import join
from os import mkdir
from shutil import rmtree
from point_viz.converter import PointvizConverter
import logging
from tqdm import tqdm
from data.utils.normalization import convert_threejs_bbox_with_colors, convert_threejs_coors

os.system("rm -r {}".format('/home/tan/tony/threejs/kitti-eval-viz'))
Converter = PointvizConverter(home='/home/tan/tony/threejs/kitti-eval-viz')

proj_home = '/home/tan/tony/dv-det'
pred_txt_home = join(proj_home, 'eval/kitti/data/txt')
img_output_home = '/home/tan/tony/threejs/kitti-eval-viz/html'


def bbox_trans(label_txt, trans_matrix_list):
    P2, R0_rect, Tr_velo_to_cam = trans_matrix_list
    inv_trans_matrix = np.linalg.inv(R0_rect.dot(Tr_velo_to_cam))
    bbox = []
    with open(label_txt, 'r') as f:
        for text_line in f.readlines():
            line = text_line.split(' ')
            label = line[0].lower()
            if label == "car":
                x, y, z, _ = inv_trans_matrix.dot(np.array([float(line[11]),
                                                            float(line[12]),
                                                            float(line[13]),
                                                            1.])).transpose().tolist()
                h = float(line[8])
                w = float(line[9])
                l = float(line[10])
                r = -float(line[14])
                z += h / 2.
                p = line[-1]
                bbox.append([w, l, h, x, y, z, r, 0, 0, p])
    return bbox


if __name__ == '__main__':
    img_npy = np.load(join(proj_home, 'dataset-all', 'img_testing.npy'))
    calib_npy = np.load(join(proj_home, 'dataset-all', 'trans_matrix_testing.npy'))
    lidar_npy = np.load(join(proj_home, 'dataset-all', 'lidar_points_testing.npy'))

    txt_file_list = os.listdir(pred_txt_home)

    for i in tqdm(range(len(lidar_npy))):
        P2, R0_rect, Tr_velo_to_cam = calib_npy[i]
        trans_matrix_list = P2.dot(R0_rect.dot(Tr_velo_to_cam))

        img = img_npy[i]
        lidar_point = lidar_npy[i]

        pred_bboxes = bbox_trans(join(pred_txt_home, txt_file_list[i]), trans_matrix_list)
        pred_bbox_params = convert_threejs_bbox_with_colors(pred_bboxes, color='red') if len(pred_bboxes) > 0 else []

        task_name = "ID_%06d_%03d" % (i, len(pred_bboxes))
        Converter.compile(task_name=task_name,
                          coors=convert_threejs_coors(lidar_point[:, :3]),
                          bbox_params=pred_bbox_params)





