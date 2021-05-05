import logging
import os
from copy import deepcopy
from os import mkdir
from os.path import join, dirname

import numpy as np
from PIL import Image
from point_viz.converter import PointvizConverter
from tqdm import tqdm

Converter = PointvizConverter(home='/home/tan/tony/threejs')
logging.basicConfig(level=logging.INFO)
CAM = 2
category_dict = {"car": 0,
                 "van": 1,
                 "truck": 1,
                 "tram": 1,
                 "misc": 1,
                 }
range_x = [0., 70.4]
range_y = [-40., 40.]
range_z = [-3., 1.]
min_object_points = 10
expand_ratio = 0.15
home = dirname(os.getcwd())


def get_union_sets(conditions):
    output = conditions[0]
    for i in np.arange(1, len(conditions)):
        output = np.logical_and(output, conditions[i])
    return output


def load_calib(calib_dir):
    lines = open(calib_dir).readlines()
    lines = [line.split()[1:] for line in lines][:-1]

    P2 = np.array(lines[CAM]).reshape(3, 4).astype('float32')

    Tr_velo_to_cam = np.array(lines[5]).reshape(3, 4).astype('float32')
    Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)

    R0_rect = np.eye(4).astype('float32')
    R0_rect[:3, :3] = np.array(lines[4][:9]).reshape(3, 3)

    return P2, R0_rect, Tr_velo_to_cam


def trim(img_dir, lidar_dir, calib_dir, range_x, range_y, range_z):
    img = np.array(Image.open(img_dir))
    rows, cols = img.shape[:2]
    img_size = [rows, cols]
    lidar_points = np.fromfile(lidar_dir, dtype=np.float32).reshape(-1, 4)  # [n, 4]
    lidar_intensity = deepcopy(lidar_points[:, -1])
    lidar_points[:, -1] = np.ones(lidar_points.shape[0])
    P2, R0_rect, Tr_velo_to_cam = load_calib(calib_dir)
    trans_matrix_list = [P2, R0_rect, Tr_velo_to_cam]
    trans_matrix = P2.dot(R0_rect.dot(Tr_velo_to_cam))

    trans_lidar_points = np.transpose(trans_matrix.dot(deepcopy(lidar_points).transpose()))  # [n, 4]
    proj_lidar_points = trans_lidar_points / trans_lidar_points[:, 2:3]
    keep_idx = get_union_sets([lidar_points[:, 0] > range_x[0],
                               lidar_points[:, 0] < range_x[1],
                               lidar_points[:, 1] > range_y[0],
                               lidar_points[:, 1] < range_y[1],
                               lidar_points[:, 2] > range_z[0],
                               lidar_points[:, 2] < range_z[1],
                               proj_lidar_points[:, 0] > 0,
                               proj_lidar_points[:, 0] < cols,
                               proj_lidar_points[:, 1] > 0,
                               proj_lidar_points[:, 1] < rows])
    trimed_lidar_points = lidar_points[keep_idx, :]
    trimed_lidar_points[:, -1] = lidar_intensity[keep_idx]
    return trimed_lidar_points, trans_matrix_list, img_size, img


def bbox_clean(label_dir, calib_dir, category_dict):
    P2, R0_rect, Tr_velo_to_cam = load_calib(calib_dir)
    inv_trans_matrix = np.linalg.inv(R0_rect.dot(Tr_velo_to_cam))
    bbox = []
    with open(label_dir, 'r') as f:
        for text_line in f.readlines():
            line = text_line.split(' ')
            label = line[0].lower()
            if label in category_dict.keys() and len(line) > 3:
                bbox_height_2d = float(line[7]) - float(line[5])
                truncation = float(line[1])
                occluded = int(line[2])
                # TODO: the difficulty split strategy can be enhanced.
                if occluded == 0 and truncation <= 0.15 and bbox_height_2d >= 40:
                    difficulty = 0
                elif occluded <= 1 and truncation <= 0.30 and bbox_height_2d >= 25:
                    difficulty = 1
                elif occluded <= 2 and truncation <= 0.50 and bbox_height_2d >= 25:
                    difficulty = 2
                elif bbox_height_2d >= 25:
                    difficulty = 3
                else:
                    difficulty = 4
                category = category_dict[label]

                x, y, z, _ = inv_trans_matrix.dot(np.array([float(line[11]),
                                                            float(line[12]),
                                                            float(line[13]),
                                                            1.])).transpose().tolist()  # from camera coors to LiDAR coors

                h = float(line[8])
                w = float(line[9])
                l = float(line[10])
                r = -float(line[14])
                z += h / 2.
                bbox.append([w, l, h, x, y, z, r, category, difficulty])
    return bbox


def plane_trans(plane_dir):
    with open(plane_dir, 'r') as f:
        for i, text_line in enumerate(f.readlines()):
            if i == 3:
                a, b, c, d = text_line.split(' ')[:4]
        return np.array([float(a), float(b), float(c), float(d)])


def get_objects(points, bboxes):
    '''
    Get the foreground points inside each valid bbox.
    :param points: the set of all the foreground points ([coors + intensity]) in each instance.
    :param bboxes: the bboxes in each instance
    :return: a list of coors of each foreground points [[n0, n1...], [n0, n1, n2...]]
    '''
    output_points = []
    output_diff = []
    output_bboxes = []
    for i in range(len(bboxes)):
        w, l, h, x, y, z, r, cls, diff = bboxes[i]
        rel_point_x = points[:, 0] - x
        rel_point_y = points[:, 1] - y
        rel_point_z = points[:, 2] - z
        rot_rel_point_x = rel_point_x * np.cos(r) + rel_point_y * np.sin(r)
        rot_rel_point_y = -rel_point_x * np.sin(r) + rel_point_y * np.cos(r)
        valid_idx = (np.abs(rot_rel_point_x) <= w * (1 + expand_ratio) / 2) * \
                    (np.abs(rot_rel_point_y) <= l * (1 + expand_ratio) / 2) * \
                    (np.abs(rel_point_z) <= h * (1 + expand_ratio) / 2)
        if np.sum(valid_idx) >= min_object_points and cls == 0:
            output_points.append(points[valid_idx])
            output_diff.append(diff)
            output_bboxes.append(bboxes[i])
    return output_points, output_diff, output_bboxes


if __name__ == '__main__':
    dataset_home = '/home/tan/tony/kitti_raw'
    output_home = join(home, 'dataset-half')
    try:
        mkdir(output_home)
    except:
        logging.warning('Directory: {} already exists.'.format(output_home))
    logging.info("Using KITTI dataset under: {}".format(dataset_home))
    for task in ['validation', 'training']:
        # for task in ['validation']:

        output_lidar_points = []
        output_bbox = []
        output_trans_matrix = []
        output_image_size = []
        output_image = []
        output_plane = []
        output_object_points = [[], [], [], [], []]
        output_object_bboxes = [[], [], [], [], []]
        img_home = join(dataset_home, 'training', 'image_2')
        lidar_home = join(dataset_home, 'training', 'velodyne')
        calib_home = join(dataset_home, 'training', 'calib')
        label_home = join(dataset_home, 'training', 'label_2')
        plane_home = join(dataset_home, 'training', 'planes')
        split_file = join(os.getcwd(), 'data_split_half', task + '.txt')
        logging.info("Processing {} dataset using split strategy: {}".format(task, split_file))

        with open(split_file, 'r') as f:
            for frame_name in tqdm(f.readlines()):
                frame_id = frame_name[:6]
                img_dir = join(img_home, '{}.png'.format(frame_id))
                lidar_dir = join(lidar_home, '{}.bin'.format(frame_id))
                calib_dir = join(calib_home, '{}.txt'.format(frame_id))
                label_dir = join(label_home, '{}.txt'.format(frame_id))
                plane_dir = join(plane_home, '{}.txt'.format(frame_id))

                lidar_points, trans_matrix, img_size, img = trim(img_dir, lidar_dir, calib_dir, range_x=range_x,
                                                                 range_y=range_y, range_z=range_z)
                plane = plane_trans(plane_dir)
                # lidar_points = length_normalize(lidar_points, length=20000)
                bbox = bbox_clean(label_dir, calib_dir, category_dict)
                points_list, diff_list, bbox_list = get_objects(points=lidar_points,
                                                                bboxes=bbox)
                if len(diff_list) > 0:
                    for j in range(len(diff_list)):
                        diff = diff_list[j]
                        output_object_points[diff].append(points_list[j])
                        output_object_bboxes[diff].append(bbox_list[j])

                output_lidar_points.append(lidar_points)
                output_bbox.append(bbox)
                output_plane.append(plane)
                output_trans_matrix.append(trans_matrix)
                output_image_size.append(img_size)
                output_image.append(img)

                # Converter.compile(task_name="prepare_lidar_points",
                #                   coors=convert_threejs_coors(lidar_points[:, :3]),
                #                   default_rgb=None,
                #                   bbox_params=None)
        logging.info("Saving...")
        np.save(join(output_home, 'lidar_points_{}.npy'.format(task)), np.array(output_lidar_points, dtype=object))
        np.save(join(output_home, 'bbox_labels_{}.npy'.format(task)), np.array(output_bbox, dtype=object))
        np.save(join(output_home, 'trans_matrix_{}.npy'.format(task)), np.array(output_trans_matrix, dtype=object))
        np.save(join(output_home, 'ground_plane_{}.npy'.format(task)), np.array(output_plane, dtype=object))
        np.save(join(output_home, 'img_size_{}.npy'.format(task)), np.array(output_image_size, dtype=object))
        np.save(join(output_home, 'img_{}.npy'.format(task)), np.array(output_image, dtype=object))
        np.save(join(output_home, "object_collections_{}.npy".format(task)), np.array(output_object_points, dtype=object))
        np.save(join(output_home, "bbox_collections_{}.npy".format(task)), np.array(output_object_bboxes, dtype=object))

    logging.info("Preprocess completed.")