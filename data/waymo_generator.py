import logging
import multiprocessing
import time
# from point_viz.utils import get_color_from_intensity
from copy import deepcopy
from os.path import join
import os
logging.basicConfig(level=logging.INFO)
import mkl
import numpy as np
from numpy.linalg import multi_dot
from point_viz.converter import PointvizConverter
from tqdm import tqdm

from data.utils.augmentation import rotate, scale, flip, drop_out, shuffle, transform, \
    get_pasted_point_cloud_waymo
from data.utils.normalization import feature_normalize, bboxes_normalization, convert_threejs_coors, \
    convert_threejs_bbox

# os.environ['MKL_NUM_THREADS'] = '1'
mkl.set_num_threads(1)

default_config = {'nbbox': 256,
                  'rotate_range': 0.,
                  'rotate_mode': 'u',
                  'scale_range': 0.,
                  'scale_mode': 'u',
                  'drop_out': 0.,
                  'flip': False,
                  'shuffle': False,
                  'paste_augmentation': False,
                  'paste_instance_num': 64,
                  'maximum_interior_points': 40,
                  'normalization': 'channel_std'}


class Dataset(object):
    def __init__(self,
                 task,
                 batch_size=16,
                 config=None,
                 queue_size=10,
                 validation=False,
                 evaluation=False,
                 random=False,
                 hvd_id=0,
                 hvd_size=1,
                 num_worker=1,
                 home='/home/tan/tony/dv-det/waymo_npy'):
        self.home = join(home, task)
        self.config = default_config if config is None else config
        self.batch_size = batch_size
        self.evaluation = evaluation
        self.validation = True if evaluation else validation
        self.task = task
        self.rotate_range = self.config['rotate_range']
        self.rotate_mode = self.config['rotate_mode']
        self.scale_range = self.config['scale_range']
        self.scale_mode = self.config['scale_mode']
        self.drop_out = self.config['drop_out']
        self.flip = self.config['flip']
        self.shuffle = self.config['shuffle']
        self.normalization = self.config['normalization']
        self.paste_augmentation = self.config['paste_augmentation']
        self.paste_instance_num = self.config['paste_instance_num']
        self.maximum_interior_points = self.config['maximum_interior_points']
        self.nbbox = self.config['nbbox']
        self.hvd_id = hvd_id
        self.random = random

        self.lidar_home = join(self.home, 'lidar')
        self.label_home = join(self.home, 'label')
        self.total_data_length = len(os.listdir(self.lidar_home))
        assert(len(os.listdir(self.lidar_home)) == len(os.listdir(self.label_home)))
        if self.hvd_id == 0:
            logging.info("Found {} files under folder: {}".format(self.total_data_length, self.lidar_home))

        # if not self.evaluation:
        #     self.bboxes = np.load(join(self.home, 'bbox_labels_{}.npy'.format(self.task)), allow_pickle=True)
        if self.config['paste_augmentation'] and not self.validation:
            self.object_collections = np.load(join(home, 'objects', 'object_collections.npy'), allow_pickle=True)
            self.bbox_collections = np.load(join(home, 'objects', 'bbox_collections.npy'), allow_pickle=True)
        #     self.ground_plane = np.load(join(self.home, 'ground_plane_{}.npy'.format(self.task)), allow_pickle=True)
        #     self.trans_matrix = np.load(join(self.home, 'trans_matrix_{}.npy'.format(self.task)), allow_pickle=True)
        # self.paste_augmentation = self.config['paste_augmentation']
        # self.paste_instance_num = self.config['paste_instance_num']
        # self.maximum_interior_points = self.config['maximum_interior_points']


        self.queue_size = queue_size
        self.num_worker = num_worker
        self.hvd_size = hvd_size
        self.hvd_data_length = self.total_data_length // self.hvd_size
        self.batch_sum = int(np.ceil(self.hvd_data_length / self.batch_size))
        self.test_start_id = self.hvd_data_length * self.hvd_id
        self.idx = self.test_start_id
        self.threads = []
        self.q = multiprocessing.Queue(maxsize=self.queue_size)
        if not self.validation:
            self.start()

    def start(self):
        for i in range(self.num_worker):
            thread = multiprocessing.Process(target=self.aug_process)
            thread.daemon = True
            self.threads.append(thread)
            thread.start()

    def stop(self):
        for i, thread in enumerate(self.threads):
            thread.terminate()
            thread.join()
            self.q.close()

    def aug_process(self):
        np.random.seed(int(time.time() * 1e3 - int(time.time()) * 1e3))
        while True:
            try:
                if self.q.qsize() < self.queue_size:
                    batch_coors = []
                    batch_features = []
                    batch_num_list = []
                    batch_bbox = np.zeros((self.batch_size, self.nbbox, 9), dtype=np.float32)

                    for i in range(self.batch_size):
                        idx = np.random.randint(self.total_data_length)

                        points = np.load(join(self.lidar_home, "%06d.npy" % idx), allow_pickle=True).astype(np.float32)
                        bboxes = np.load(join(self.label_home, "%06d.npy" % idx), allow_pickle=True).astype(np.float32)

                        if len(bboxes) > 0:
                            bboxes = bboxes[bboxes[:, 0] > 0, :]

                        if self.paste_augmentation:
                            points, bboxes = get_pasted_point_cloud_waymo(scene_points=points,
                                                                          scene_bboxes=bboxes,
                                                                          object_collections=self.object_collections,
                                                                          bbox_collections=self.bbox_collections,
                                                                          instance_num=self.paste_instance_num,
                                                                          maximum_interior_points=self.maximum_interior_points)
                        # print(points.shape, bboxes.shape)
                        if self.shuffle:
                            points = shuffle(points)
                        if self.drop_out > 0:
                            points = drop_out(points, self.drop_out)
                        coors = points[:, :3]
                        features = points[:, -2:]

                        T_rotate, angle = rotate(self.rotate_range, self.rotate_mode)
                        T_scale, scale_xyz = scale(self.scale_range, self.scale_mode)
                        T_flip, flip_y = flip(flip=self.flip)

                        T_coors = multi_dot([T_scale, T_flip, T_rotate])
                        coors = transform(coors, T_coors)

                        # keep_idx = range_clip(coors, self.range_x, self.range_y, self.range_z)
                        # coors = coors[keep_idx, :]
                        # features = features[keep_idx, :]

                        features = feature_normalize(features, method=self.normalization)
                        ret_bboxes = []
                        for box in bboxes:

                            w, l, h, x, y, z, r = box[:7]
                            x, y, z = transform(np.array([x, y, z]), T_coors)
                            w, l, h = transform(np.array([w, l, h]), T_scale)
                            r += angle
                            if flip_y == -1:
                                r = (-1) ** int(r <= 0) * np.pi - r
                            if np.abs(r) > 2 * np.pi:
                                r = np.abs(r) % (2 * np.pi) * (-1) ** int(r <= 0)
                            if np.abs(r) > np.pi:
                                r = -(2 * np.pi - np.abs(r))

                            category = box[-2]
                            difficulty = box[-1]
                            ret_bboxes.append([w, l, h, x, y, z, r, category, difficulty])
                        bboxes = bboxes_normalization(ret_bboxes, length=self.nbbox)

                        batch_coors.append(coors)
                        batch_features.append(features)
                        batch_num_list.append(len(coors))
                        batch_bbox[i] = bboxes

                        # print(coors.shape, features.shape, bboxes.shape)

                    batch_coors = np.concatenate(batch_coors, axis=0)
                    batch_features = np.concatenate(batch_features, axis=0)
                    batch_num_list = np.array(batch_num_list)
                    batch_bbox = np.array(batch_bbox)
                    self.q.put([batch_coors, batch_features, batch_num_list, batch_bbox])
                    # print(self.q.qsize())
                else:
                    # print(self.q.qsize(), "Sleep for 0.05s.")
                    time.sleep(0.05)
            except:
                self.stop()

    def train_generator(self):
        while True:
            if self.q.qsize() > 0:
                yield self.q.get()
            else:
                time.sleep(0.05)

    def valid_generator(self, start_idx=None):
        if start_idx is not None:
            self.idx = start_idx
        while True:
            stop_idx = int(np.min([self.idx + self.batch_size, self.hvd_data_length * (self.hvd_id + 1)]))
            batch_size = stop_idx - self.idx
            batch_coors = []
            batch_features = []
            batch_num_list = []
            batch_bbox = np.zeros((batch_size, self.nbbox, 9), dtype=np.float32)
            for i in range(batch_size):
                idx = np.random.randint(self.total_data_length) if self.random else self.idx
                points = np.load(join(self.lidar_home, "%06d.npy" % idx), allow_pickle=True).astype(np.float32)
                coors = points[:, :3]
                features = points[:, -2:]

                if len(coors) == 0:
                    coors = np.array([[1., 0., 0.]])  # to keep the npoint always > 0 in a frame
                    features = np.array([[0., 0.]])

                # keep_idx = range_clip(coors, self.range_x, self.range_y, self.range_z)
                # coors = coors[keep_idx, :]
                # features = features[keep_idx, :]

                batch_coors.append(coors)
                batch_features.append(feature_normalize(features, method=self.normalization))
                batch_num_list.append(len(coors))
                if not self.evaluation:
                    bboxes = np.load(join(self.label_home, "%06d.npy" % idx), allow_pickle=True).astype(np.float32)
                    if len(bboxes) > 0:
                        bboxes = bboxes[bboxes[:, 0] > 0, :]
                    batch_bbox[i] = bboxes_normalization(bboxes, length=self.nbbox)
                self.idx += 1
            self.idx = self.test_start_id if stop_idx == self.hvd_data_length * (self.hvd_id + 1) else stop_idx
            batch_coors = np.concatenate(batch_coors, axis=0)
            batch_features = np.concatenate(batch_features, axis=0)
            batch_num_list = np.array(batch_num_list)
            batch_bbox = np.array(batch_bbox)
            if self.evaluation:
                yield batch_coors, batch_features, batch_num_list
            else:
                yield batch_coors, batch_features, batch_num_list, batch_bbox


if __name__ == '__main__':
    aug_config = {'nbbox': 256,
                  'rotate_range': np.pi * 2,
                  'rotate_mode': 'u',
                  'scale_range': 0.05,
                  'scale_mode': 'u',
                  'drop_out': 0.1,
                  'flip': False,
                  'shuffle': True,
                  'paste_augmentation': True,
                  'paste_instance_num': 128,
                  'maximum_interior_points': 40,
                  'normalization': 'channel_std'}

    dataset = Dataset(task='train',
                      config=aug_config,
                      batch_size=4,
                      validation=False,
                      num_worker=20,
                      hvd_size=8,
                      hvd_id=0)
    generator = dataset.train_generator()
    for i in tqdm(range(10000)):
        # dataset.aug_process()
        coors, features, num_list, bboxes = next(generator)



        # print(num_list)
        # print(coors.shape, features.shape, num_list)

        dimension = [200., 200., 8.]
        offset = [100., 100., 3.0]

        coors += offset
        coors_min = np.min(coors, axis=0)
        coors_max = np.max(coors, axis=0)
        # print(coors_min, coors_max)
        for j in range(3):
            if coors_min[j] < 0 or coors_max[j] > dimension[j]:
                print("***", coors_min, coors_max)

    # coors, ref, attention, bboxes = next(dataset.train_generator())
    # dataset.stop()
    batch_id = 2
    acc_num_list = np.cumsum(num_list)
    #
    coors = coors[acc_num_list[batch_id-1]:acc_num_list[batch_id], :]
    features = features[acc_num_list[batch_id-1]:acc_num_list[batch_id], :]
    bboxes = bboxes[batch_id]

    Converter = PointvizConverter(home='/home/tan/tony/threejs')
    Converter.compile(task_name="waymo_generator",
                      coors=convert_threejs_coors(coors),
                      intensity=features[:, 0],
                      default_rgb=None,
                      bbox_params=convert_threejs_bbox(bboxes))
