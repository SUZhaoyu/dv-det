import numpy as np
import cv2
import os
from os.path import join
from os import mkdir
from shutil import rmtree
from point_viz.converter import PointvizConverter
import logging
from copy import deepcopy
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
    bboxes = []
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
                p = float(line[-1])
                bboxes.append([w, l, h, x, y, z, r, 0, 0, p])
    return bboxes


def draw_2d_bbox(img, bboxes, trans_matrix_list):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    P2, R0_rect, Tr_velo_to_cam = trans_matrix_list
    trans_matrix = P2.dot(R0_rect.dot(Tr_velo_to_cam))
    img_h, img_w = img.shape[:2]
    for bbox in bboxes:
        w, l, h, x, y, z, r = bbox[:7]  # Lidar coor
        conf = bbox[-1]
        p0 = [+w / 2, +l / 2, +h / 2]
        p1 = [+w / 2, +l / 2, -h / 2]
        p2 = [+w / 2, -l / 2, +h / 2]
        p3 = [+w / 2, -l / 2, -h / 2]
        p4 = [-w / 2, +l / 2, +h / 2]
        p5 = [-w / 2, +l / 2, -h / 2]
        p6 = [-w / 2, -l / 2, +h / 2]
        p7 = [-w / 2, -l / 2, -h / 2]
        v = np.array([p0, p1, p2, p3, p4, p5, p6, p7])  # [8, 3]
        rotate_matrix = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        v[:, :2] = v[:, :2].dot(rotate_matrix.transpose())
        v += [x, y, z]
        v = np.concatenate([v, np.ones(shape=[8, 1])], axis=-1)  # [8, 4]
        proj_v = v.dot(trans_matrix.transpose())
        proj_v /= proj_v[:, 2:3]
        # proj_v[:, 0] = np.clip(proj_v[:, 0], 0, img_w - 1)
        # proj_v[:, 1] = np.clip(proj_v[:, 1], 0, img_h - 1)
        proj_v = np.array(proj_v[:, :2], dtype=np.int32)
        # text_loc = (proj_v[0, :2] + proj_v[4, :2]) / 2. - [5, 0]
        proj_v = tuple(map(tuple, proj_v))

        cv2.line(img, proj_v[0], proj_v[2], (255, 255, 0), 1)
        cv2.line(img, proj_v[1], proj_v[3], (255, 255, 0), 1)
        cv2.line(img, proj_v[5], proj_v[7], (255, 255, 0), 1)
        cv2.line(img, proj_v[6], proj_v[4], (255, 255, 0), 1)

        cv2.line(img, proj_v[0], proj_v[4], (0, 255, 0), 1)
        cv2.line(img, proj_v[0], proj_v[1], (0, 255, 0), 1)
        cv2.line(img, proj_v[1], proj_v[5], (0, 255, 0), 1)
        cv2.line(img, proj_v[5], proj_v[4], (0, 255, 0), 1)

        cv2.line(img, proj_v[2], proj_v[3], (0, 0, 255), 1)
        cv2.line(img, proj_v[2], proj_v[6], (0, 0, 255), 1)
        cv2.line(img, proj_v[6], proj_v[7], (0, 0, 255), 1)
        cv2.line(img, proj_v[3], proj_v[7], (0, 0, 255), 1)

        # cv2.putText(img, str(conf), (int(text_loc[0]), int(text_loc[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


    return img



if __name__ == '__main__':
    img_npy = np.load(join(proj_home, 'dataset', 'img_validation.npy'), allow_pickle=True)
    calib_npy = np.load(join(proj_home, 'dataset', 'trans_matrix_validation.npy'), allow_pickle=True)
    lidar_npy = np.load(join(proj_home, 'dataset', 'lidar_points_validation.npy'), allow_pickle=True)

    txt_file_list = os.listdir(pred_txt_home)

    for i in tqdm(range(len(lidar_npy))):
        trans_matrix_list = calib_npy[i]

        img = deepcopy(img_npy[i]).astype(np.uint8)
        lidar_point = lidar_npy[i]

        pred_bboxes = bbox_trans(join(pred_txt_home, txt_file_list[i]), trans_matrix_list)
        img = draw_2d_bbox(img, pred_bboxes, trans_matrix_list)
        pred_bbox_params = convert_threejs_bbox_with_colors(pred_bboxes, color='red') if len(pred_bboxes) > 0 else []

        task_name = "ID_%06d_%03d" % (i, len(pred_bboxes))
        Converter.compile(task_name=task_name,
                          coors=convert_threejs_coors(lidar_point[:, :3]),
                          bbox_params=pred_bbox_params)

        cv2.imwrite(join(img_output_home, task_name + '.png'), img)
