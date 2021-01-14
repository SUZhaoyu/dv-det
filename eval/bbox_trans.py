import logging
from os import mkdir
from os.path import join
from shutil import rmtree

import numpy as np
from tqdm import tqdm


def get_bbox_center(input_bbox):
    output_center = []
    for box in input_bbox:
        x, y, z = box[3:6]
        z -= box[2] / 2
        output_center.append([x, y, z]) # [n, 3]
    output_center = np.array(output_center) # [n, 3]
    return output_center # LiDAR coor

def get_2d_bbox(input_bbox, trans_matrix_list, img_size):
    P2, R0_rect, Tr_velo_to_cam = trans_matrix_list
    trans_matrix = P2.dot(R0_rect.dot(Tr_velo_to_cam))
    row, col = img_size
    output_2d_bbox = []
    output_3d_bbox = []
    for box in input_bbox:
        w, l, h, x, y, z, r = box[:7] # Lidar coor
        p0 = [+w/2, +l/2, +h/2]
        p1 = [+w/2, +l/2, -h/2]
        p2 = [+w/2, -l/2, +h/2]
        p3 = [+w/2, -l/2, -h/2]
        p4 = [-w/2, +l/2, +h/2]
        p5 = [-w/2, +l/2, -h/2]
        p6 = [-w/2, -l/2, +h/2]
        p7 = [-w/2, -l/2, -h/2]
        v = np.array([p0, p1, p2, p3, p4, p5, p6, p7]) # [8, 3]
        rotate_matrix = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        v[:, :2] = v[:, :2].dot(rotate_matrix.transpose())
        v += [x, y, z]
        v = np.concatenate([v, np.ones(shape=[8, 1])], axis=-1) # [8, 4]
        proj_v = v.dot(trans_matrix.transpose())
        proj_v /= proj_v[:, 2:3]
        if np.min(proj_v[:, 0]) < 0. or \
           np.max(proj_v[:, 0]) > col or \
           np.min(proj_v[:, 1]) < 0. or \
           np.max(proj_v[:, 1]) > row:
            continue
        else:
            left, right = np.min(proj_v[:, 0]), np.max(proj_v[:, 0])
            top, bottom = np.min(proj_v[:, 1]), np.max(proj_v[:, 1])
            output_2d_bbox.append([left, top, right, bottom])
            output_3d_bbox.append(box)

    return np.array(output_2d_bbox), np.array(output_3d_bbox) # [n, 8, 3], LiDAR coor

def get_center_coors(input_bbox, trans_matrix_list):
    P2, R0_rect, Tr_velo_to_cam = trans_matrix_list
    trans_matrix = R0_rect.dot(Tr_velo_to_cam)
    input_bbox[:, 5] -= input_bbox[:, 2] / 2.
    center_coors = np.concatenate([input_bbox[:, 3:6], np.ones(shape=[len(input_bbox), 1])], axis=-1)

    return center_coors.dot(trans_matrix.transpose())[:, :3]


def write_txt(txt_dir, bbox_2d, center_coors, input_bbox, category='Car'):
    valid_bbox_num = 0
    for i in range(len(bbox_2d)):
        type = category
        truncation = ' -1'
        occlusion = ' -1'
        alpha = ' -10'
        x1 = ' %.2f'%bbox_2d[i, 0]
        y1 = ' %.2f'%bbox_2d[i, 1]
        x2 = ' %.2f'%bbox_2d[i, 2]
        y2 = ' %.2f'%bbox_2d[i, 3]
        w = ' %.2f'%min(input_bbox[i, 0], input_bbox[i, 1])
        l = ' %.2f'%max(input_bbox[i, 0], input_bbox[i, 1])
        h = ' %.2f'%input_bbox[i, 2]
        x_c = ' %.2f'%center_coors[i, 0]
        y_c = ' %.2f'%center_coors[i, 1]
        z_c = ' %.2f'%center_coors[i, 2]
        score = ' %.2f\n'%input_bbox[i, -1]
        # score = ' %.2f\n'%0.95
        r = -input_bbox[i, 6]
        if np.abs(r) > np.pi:
            r = (2 * np.pi - np.abs(r)) * ((-1) ** (r // np.pi))
        r = ' %.2f' % r
        bbox_height = float(y2) - float(y1)
        if bbox_height > 25:
            text = type + truncation + occlusion + alpha + x1 + y1 + x2 + y2 + h + w + l + x_c + y_c + z_c + r + score
            with open(txt_dir, 'a') as f:
                f.write(text)
            valid_bbox_num += 1
    return valid_bbox_num

def write_null_txt(txt_dir):
    with open(txt_dir, 'a') as f:
        f.write('DontCare -1 -1 -10 -1 -1 -1 -1 -1 -1 -1 -1000 -1000 -1000 -10')


if __name__ == '__main__':
    home = '/home/tan/tony/dv-det'
    calib_home = join(home, 'dataset')
    prediction_home = join(home, 'eval', 'data')
    output_txt_home = join(home, 'eval', 'txt')
    logging.info("Using KITTI dataset under: {}".format(home))
    TASK = 'validation'

    try: rmtree(output_txt_home)
    except: pass

    try: mkdir(output_txt_home)
    except: logging.warning('Directory: {} already exists.'.format(output_txt_home))

    input_bbox_predictions = np.load(join(prediction_home, 'bbox_predictions.npy'), allow_pickle=True)
    # input_bbox_predictions = np.load(join(prediction_home, 'bbox_testing.npy'), allow_pickle=True)
    input_trans_matrix = np.load(join(calib_home, 'trans_matrix_{}.npy'.format(TASK)), allow_pickle=True)
    input_image_size = np.load(join(calib_home, 'img_size_{}.npy'.format(TASK)), allow_pickle=True)

    for i in tqdm(range(len(input_bbox_predictions))):
        bbox_labels = np.array(input_bbox_predictions[i])
        trans_matrix = input_trans_matrix[i]
        img_size = input_image_size[i]
        txt_dir = join(output_txt_home, "%06d.txt" % int(i))
        valid_bbox_num = 0
        bbox_2d, bbox_3d = [], []

        if len(bbox_labels) > 0:
            bbox_2d, bbox_3d = get_2d_bbox(bbox_labels, trans_matrix, img_size)

        if len(bbox_3d) > 0:
            center_coors = get_center_coors(bbox_3d, trans_matrix)
            valid_bbox_num = write_txt(txt_dir, bbox_2d, center_coors, bbox_3d)

        if valid_bbox_num == 0:
            write_null_txt(txt_dir)
