from os.path import join
from os import mkdir
import logging
from tqdm import tqdm
import numpy as np
import sys
sys.path.append('/home/tan/tony/dv-det')
from random import shuffle
from point_viz.converter import PointvizConverter
logging.basicConfig(level=logging.INFO)
from data.utils.normalization import get_union_sets, convert_threejs_coors, convert_threejs_bbox
Converter = PointvizConverter("/home/tan/tony/threejs")


task = 'val'
keep_ratio = 0.2
scene_num_dict = {'train': 798,
                  'val': 202}
range_x = [-75.2, 75.2]
range_y = [-75.2, 75.2]
range_z = [-2., 4.]

expand_ratio = 0.15
min_object_points = 15
min_vehicle_length = 3.0


raw_home = join('/media/data1/waymo_npy_from_tfrecord', task)
output_home = '/media/data1/waymo_npy_from_tfrecord/segments'
lidar_output_home = join(output_home, '{}/lidar'.format(task))
label_output_home = join(output_home, '{}/label'.format(task))


scene_num = scene_num_dict[task]
output_frame_id_list = []

def split(a, n):
    k, m = divmod(len(a), n)
    return list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def trim(lidar_points):
    keep_idx = get_union_sets([lidar_points[:, 0] > range_x[0],
                               lidar_points[:, 0] < range_x[1],
                               lidar_points[:, 1] > range_y[0],
                               lidar_points[:, 1] < range_y[1],
                               lidar_points[:, 2] > range_z[0],
                               lidar_points[:, 2] < range_z[1],
                               lidar_points[:, -1] < 0])
    trimed_lidar_points = lidar_points[keep_idx, :]
    return trimed_lidar_points

def bbox_clean(input_bboxes):
    output_bboxes = []
    for bbox in input_bboxes:
        if bbox[1] == 1 and bbox[11] <= 2 and bbox[9] > 0:
            x, y, z, l, w, h, r = bbox[2:9]
            diff = bbox[11]
            output_bboxes.append([w, l, h, x, y, z, r + np.pi / 2, 0, diff])
    output_bboxes = np.array(output_bboxes)

    return output_bboxes

def get_objects(points, bboxes):
    '''
    Get the foreground points inside each valid bbox.
    :param points: the set of all the foreground points ([coors + intensity]) in each instance.
    :param bboxes: the bboxes in each instance
    :return: a list of coors of each foreground points [[n0, n1...], [n0, n1, n2...]]
    '''
    output_points = []
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
        if np.sum(valid_idx) > min_object_points and l > min_vehicle_length:
            output_points.append(points[valid_idx])
            output_bboxes.append(bboxes[i])
    return output_points, output_bboxes


if __name__ == '__main__':
    scene_id_list = np.arange(scene_num_dict[task]).tolist()
    logging.info("Process with keep ratio {}".format(keep_ratio))
    output_lidar = []
    output_label = []
    output_object_lidar = []
    output_object_label = []
    frame_count = 0
    logging.info("Loading raw dataset...")
    for scene_id in tqdm(scene_id_list):
        scene_lidar = np.load(join(raw_home, '{}_lidars_{}.npy'.format(task, scene_id)), allow_pickle=True)
        scene_label = np.load(join(raw_home, '{}_output_labels_{}.npy'.format(task, scene_id)), allow_pickle=True)
        assert len(scene_lidar) == len(scene_label)
        scene_frame_num = len(scene_label)
        selected_scene_frame_num = int(np.floor(scene_frame_num * keep_ratio))
        for i in range(selected_scene_frame_num):
            frame_id = int(i / keep_ratio)
            frame_lidar = scene_lidar[frame_id].astype(np.float32)
            frame_label = scene_label[frame_id].astype(np.float32)
            frame_lidar = trim(frame_lidar)
            frame_label = bbox_clean(frame_label)

            if task == 'train':
                points_list, bbox_list = get_objects(points=frame_lidar,
                                                     bboxes=frame_label)
                if len(bbox_list) > 0:
                    for j in range(len(bbox_list)):
                        output_object_lidar.append(points_list[j])
                        output_object_label.append(bbox_list[j])


            np.save(join(lidar_output_home, "%06d.npy" % frame_count), np.array(frame_lidar, dtype=object))
            np.save(join(label_output_home, "%06d.npy" % frame_count), np.array(frame_label, dtype=object))
            frame_count += 1
    if task == 'train':
        np.save(join(output_home, "objects", "object_collections.npy"), np.array(output_object_lidar, dtype=object))
        np.save(join(output_home, "objects", "bbox_collections.npy"), np.array(output_object_label, dtype=object))

    logging.info("Completed.")


