from os.path import join
from os import mkdir
import logging
from tqdm import tqdm
import numpy as np
from random import shuffle
from point_viz.converter import PointvizConverter
logging.basicConfig(level=logging.INFO)
from data.utils.normalization import get_union_sets, convert_threejs_coors, convert_threejs_bbox
Converter = PointvizConverter("/home/tan/tony/threejs")

task = 'val'
node_num = 16
keep_ratio = 0.2
scene_num_dict = {'train': 798,
                  'val': 202}
range_x = [-75.2, 75.2]
range_y = [-75.2, 75.2]
range_z = [-2., 4.]

raw_home = join('/media/data1/waymo', task)
lidar_output_home = join('/media/data1/waymo/segments', '{}/lidar'.format(task))
label_output_home = join('/media/data1/waymo/segments', '{}/label'.format(task))


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
    # if len(output_bboxes.shape) == 1 and output_bboxes.shape[0] != 0:
    #     output_bboxes = np.expand_dims(output_bboxes, axis=0)
    # if len(output_bboxes.shape) == 1 and output_bboxes.shape[0] == 0:
    #     output_bboxes = np.array([[0.1, 0.1, 0.1, 0., 0., 0., 0., 2]])

    return output_bboxes

total_output_label = []

if __name__ == '__main__':
    scene_id_list = np.arange(scene_num_dict[task]).tolist()
    logging.info("Process with keep ratio {}".format(keep_ratio))
    output_lidar = []
    output_label = []
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
            frame_lidar = scene_lidar[frame_id]
            frame_label = scene_label[frame_id]
            frame_lidar = trim(frame_lidar)
            frame_label = bbox_clean(frame_label)
            np.save(join(lidar_output_home, "%06d.npy" % frame_count), np.array(frame_lidar, dtype=object))
            np.save(join(label_output_home, "%06d.npy" % frame_count), np.array(frame_label, dtype=object))
            frame_count += 1

        # if scene_id > 10:
        #     break

    logging.info("Completed.")







