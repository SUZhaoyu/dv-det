import logging
import os
from copy import deepcopy
from os import mkdir
from os.path import join, dirname
logging.basicConfig(level=logging.INFO)
import numpy as np
from PIL import Image
from point_viz.converter import PointvizConverter
from tqdm import tqdm

from data.utils.normalization import get_union_sets

Converter = PointvizConverter(home='/home/tan/tony/threejs')
logging.basicConfig(level=logging.INFO)
CAM = 2

range_x = [0., 70.4]
range_y = [-40., 40.]
range_z = [-3., 1.]
min_object_points = 15
expand_ratio = 0.15
home = dirname(os.getcwd())


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


if __name__ == '__main__':
    # dataset_home = '/home/tan/tony/kitti_raw'
    dataset_home = '/media/data1/kitti_raw'
    output_home = join(home, 'dataset-eval')
    try:
        mkdir(output_home)
    except:
        logging.warning('Directory: {} already exists.'.format(output_home))
    logging.info("Using KITTI dataset under: {}".format(dataset_home))
    task = 'testing'
            # for task in ['validation']:

    output_lidar_points = []
    output_trans_matrix = []
    output_image_size = []
    output_image = []
    file_names = []
    img_home = join(dataset_home, 'testing', 'image_2')
    lidar_home = join(dataset_home, 'testing', 'velodyne')
    calib_home = join(dataset_home, 'testing', 'calib')
    split_file = join(os.getcwd(), 'data_split_eval', task + '.txt')
    logging.info("Processing {} dataset using split strategy: {}".format(task, split_file))

    with open(split_file, 'r') as f:
        for frame_name in tqdm(f.readlines()):
            frame_id = frame_name[:6]
            img_dir = join(img_home, '{}.png'.format(frame_id))
            lidar_dir = join(lidar_home, '{}.bin'.format(frame_id))
            calib_dir = join(calib_home, '{}.txt'.format(frame_id))

            lidar_points, trans_matrix, img_size, img = trim(img_dir, lidar_dir, calib_dir, range_x=range_x,
                                                             range_y=range_y, range_z=range_z)


            output_lidar_points.append(lidar_points)
            output_trans_matrix.append(trans_matrix)
            output_image_size.append(img_size)
            output_image.append(img)
            file_names.append(frame_id)

            # Converter.compile(task_name="prepare_lidar_points",
            #                   coors=convert_threejs_coors(lidar_points[:, :3]),
            #                   default_rgb=None,
            #                   bbox_params=None)
    logging.info("Saving...")
    np.save(join(output_home, 'lidar_points_{}.npy'.format(task)), np.array(output_lidar_points, dtype=object))
    np.save(join(output_home, 'trans_matrix_{}.npy'.format(task)), np.array(output_trans_matrix, dtype=object))
    np.save(join(output_home, 'img_size_{}.npy'.format(task)), np.array(output_image_size, dtype=object))
    np.save(join(output_home, 'img_{}.npy'.format(task)), np.array(output_image, dtype=object))
    np.save(join(output_home, 'file_names_{}.npy'.format(task)), np.array(file_names, dtype=object))

    logging.info("Preprocess completed.")
