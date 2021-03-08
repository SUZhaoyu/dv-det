from os.path import join
from random import choice

import numpy as np
from point_viz.converter import PointvizConverter
from tqdm import tqdm

Converter = PointvizConverter(home='/home/tan/tony/threejs/test')

data_home = '/home/tan/tony/dv-det/eval/data'
input_bboxes_stack = np.load(join(data_home, 'input_bboxes.npy'), allow_pickle=True)

if __name__ == '__main__':
    testing_output = []
    for i in tqdm(range(len(input_bboxes_stack))):
        output_bboxes = input_bboxes_stack[i][0]
        output_bboxes = output_bboxes[output_bboxes[:, 0] != 0, :]
        w = output_bboxes[:, 0] * ((np.random.rand() * 0.1 - 0.05) + 1.)
        l = output_bboxes[:, 1] * ((np.random.rand() * 0.1 - 0.05) + 1.)
        h = output_bboxes[:, 2] * ((np.random.rand() * 0.1 - 0.05) + 1.)
        x = output_bboxes[:, 3] + (np.random.rand() * 0.2 - 0.1)
        y = output_bboxes[:, 4] + (np.random.rand() * 0.2 - 0.1)
        z = output_bboxes[:, 5] + (np.random.rand() * 0.2 - 0.1)
        r = output_bboxes[:, 6] * ((np.random.rand() * 0.2 - 0.1) + 1.)

        # w = output_bboxes[:, 0]
        # l = output_bboxes[:, 1]
        # h = output_bboxes[:, 2]
        # x = output_bboxes[:, 3]
        # y = output_bboxes[:, 4]
        # z = output_bboxes[:, 5]
        # r = output_bboxes[:, 6]

        if np.random.rand() > 0.5:
            w, l = l, w
            r += np.pi * choice([-0.5, 0.5])


        c = np.zeros(len(w))
        d = np.zeros(len(w))
        # p = np.clip(np.random.randn(len(w)) * 0.2 + 0.9, 0., 1.0)
        p = np.random.rand(len(w))
        bboxes = np.stack([w, l, h, x, y, z, r, c, d, p], axis=-1)
        # if np.random.rand() < 0.2:
        #     bboxes = np.vstack((bboxes, np.array([1.5, 3., 1.6, 10., -10., 5, 0., 0., 0., np.random.rand()])))
        testing_output.append(bboxes)
    np.save('/home/tan/tony/dv-det/eval/data/bbox_testing.npy', testing_output)




