from __future__ import division
from copy import deepcopy

# import cv2
import numpy as np
from numpy.linalg import multi_dot
from shapely.geometry import Polygon

'''
This augmentation is performed instance-wised, not applicable to batch-processing.

'''


def shuffle(data):
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    return data[idx, :]


def gauss_dist(mean, shift):
    if shift != 0:
        std = shift * 0.5
        return_value = np.clip(np.random.normal(mean, std), mean - shift, mean + shift)
        return return_value
    else:
        return mean


def uni_dist(mean, shift):
    if shift != 0:
        return_value = (np.random.rand() - 0.5) * 2 * shift + mean
        return return_value
    else:
        return mean


def scale(scale_range, mode='g', scale_xyz=None, T=None):
    if T is None:
        T = np.eye(3)
    if scale_xyz is None:
        if mode == 'g':
            scale_factor_xy = gauss_dist(1., scale_range)
            scale_factor_z = gauss_dist(1., scale_range)
        elif mode == 'u':
            scale_factor_xy = uni_dist(1., scale_range)
            scale_factor_z = uni_dist(1., scale_range)
        else:
            raise ValueError("Undefined scale mode: {}".format(mode))
    else:
        scale_factor_x, scale_factor_y, scale_factor_z = scale_xyz
    T = np.dot(T, np.array([[scale_factor_xy, 0, 0],
                            [0, scale_factor_xy, 0],
                            [0, 0, scale_factor_z]]))
    return T, [scale_factor_xy, scale_factor_z]


def flip(flip=False, T=None):
    if T is None:
        T = np.eye(3)
    if not flip:
        return np.dot(T, np.eye(3)), 1.
    else:
        flip_y = -1 if np.random.rand() > 0.5 else 1
        T = np.dot(T, np.array([[1, 0, 0],
                                [0, flip_y, 0],
                                [0, 0, 1]]))
        return T, flip_y


def rotate(rotate_range, mode, angle=None, T=None):
    if T is None:
        T = np.eye(3)
    if angle is None:
        if mode == 'g':
            angle = gauss_dist(0., rotate_range)
        elif mode == 'u':
            angle = np.random.uniform(-rotate_range, rotate_range)
        else:
            raise ValueError("Undefined rotate mode: {}".format(mode))

    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])
    T = multi_dot([R, T])
    return T, angle


def shear(shear_range, mode, shear_xy=None, T=None):
    if T is None:
        T = np.eye(3)
    # TODO: Need to change the angles_z into uniform_dist for ModelNet_40
    if shear_xy is None:
        if mode == 'g':
            lambda_x = gauss_dist(0., shear_range)
            lambda_y = gauss_dist(0., shear_range)
        elif mode == 'u':
            lambda_x = np.random.uniform(0., shear_range)
            lambda_y = np.random.uniform(0., shear_range)
        else:
            raise ValueError("Undefined shear mode: {}".format(mode))
    else:
        lambda_x, lambda_y = shear_xy
    Sx = np.array([[1, 0, lambda_x],
                   [0, 1, 0],
                   [0, 0, 1]])
    Sy = np.array([[1, 0, 0],
                   [0, 1, lambda_y],
                   [0, 0, 1]])
    T = multi_dot([Sx, Sy, T])
    return T, [lambda_x, lambda_y]


def transform(data, T):
    transformed = np.transpose(np.dot(T, np.transpose(data)))
    return transformed


def drop_out(data, drop_rate):
    assert len(data.shape) == 2
    assert 0. <= drop_rate < 0.5
    if drop_rate != 0:
        real_drop_rate = np.random.uniform(low=1. - drop_rate, high=1.)
        length = int(np.ceil(len(data) * real_drop_rate))
        selected_idxs = np.random.choice(range(len(data)), length, replace=False)
        data = data[selected_idxs]
    return data


def ones_padding(raw_input):
    features = np.ones(shape=(raw_input.shape[0], raw_input.shape[1], 1), dtype=np.float32)
    return features


# def perspective_transformation(map, scale, padding_list=None, mode='g'):
#     assert len(map.shape) == 3
#     height = map.shape[0]
#     width = map.shape[1]
#     channels = map.shape[-1]
#     if padding_list is None:
#         padding_list = np.zeros(channels)
#     else:
#         assert len(padding_list) == channels
#     if mode == 'g':
#         rand_method = gauss_dist
#     elif mode == 'u':
#         rand_method = uni_dist
#     else:
#         raise NameError("Unsupported random mode: {}".format(mode))
#     original_corners = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
#     new_corners = np.float32([[rand_method(0, scale * width), rand_method(0, scale * height)],
#                               [rand_method(width, scale * width), rand_method(0, scale * height)],
#                               [rand_method(0, scale * width), rand_method(height, scale * height)],
#                               [rand_method(width, scale * width), rand_method(height, scale * height)]])
#     M = cv2.getPerspectiveTransform(new_corners, original_corners)
#     for c in range(channels):
#         map[:, :, c] = cv2.warpPerspective(map[:, :, c], M, (width, height),
#                                            flags=cv2.INTER_NEAREST,
#                                            borderMode=cv2.BORDER_CONSTANT,
#                                            borderValue=padding_list[c])
#
#     return map


def horizontal_flip(map):
    return map[:, ::-1, :]


def img_feature_value_perturbation(map, noise_scale=0, offset_scale=0, depth_scale=0, mode='g'):
    assert len(map.shape) == 3
    channels = map.shape[-1]
    height = map.shape[0]
    width = map.shape[1]
    mask = map[:, :, 0] != -1
    depth_span = np.percentile(map[mask, 0], 95) - np.percentile(map[mask, 0], 5)
    for c in range(channels):
        std = np.std(map[mask, c])
        span = np.percentile(map[mask, c], 95) - np.percentile(map[mask, c], 5)
        if mode == 'g':
            noise = np.random.randn(height, width) * std * noise_scale
            offset = gauss_dist(mean=0, shift=offset_scale * span)
            depth_wise = gauss_dist(mean=0, shift=span * depth_scale) * map[mask, 0] / depth_span
        elif mode == 'u':
            noise = (np.random.rand(height, width) - 0.5) * 2 * std * noise_scale
            offset = uni_dist(mean=0, shift=offset_scale * span)
            depth_wise = uni_dist(mean=0, shift=span * depth_scale) * map[mask, 0] / depth_span
        else:
            raise NameError("Unsupported random mode: {}".format(mode))
        map[mask, c] = map[mask, c] + noise[mask] + offset + depth_wise

    return map


def point_feature_value_perturbation(features, depth, noise_scale=0, offset_scale=0, depth_scale=0, mode='g'):
    assert len(features.shape) == 2
    assert len(depth.shape) == 1
    assert features.shape[0] == depth.shape[0]
    channels = features.shape[-1]
    npoint = features.shape[0]
    depth_span = np.percentile(depth, 95) - np.percentile(depth, 5)
    for c in range(channels):
        std = np.std(features[:, c])
        span = np.percentile(features[:, c], 95) - np.percentile(features[:, c], 5)
        if mode == 'g':
            noise = np.random.randn(npoint) * std * noise_scale
            offset = gauss_dist(mean=0, shift=offset_scale * span)
            depth_wise = gauss_dist(mean=0, shift=span * depth_scale) * depth / depth_span
        elif mode == 'u':
            noise = (np.random.rand(npoint) - 0.5) * 2 * std * noise_scale
            offset = uni_dist(mean=0, shift=offset_scale * span)
            depth_wise = uni_dist(mean=0, shift=span * depth_scale) * depth / depth_span
        else:
            raise NameError("Unsupported random mode: {}".format(mode))
        features[:, c] = features[:, c] + noise + offset + depth_wise

    return features


def get_polygon_from_bbox(bbox, expend_ratio=0.):
    w, l, x, y, r = bbox[0], bbox[1], bbox[3], bbox[4], bbox[6]
    rel_vertex = np.array([[(w + w * expend_ratio) / 2, (l + l * expend_ratio) / 2],
                           [-(w + w * expend_ratio) / 2, (l + l * expend_ratio) / 2],
                           [-(w + w * expend_ratio) / 2, -(l + l * expend_ratio) / 2],
                           [(w + w * expend_ratio) / 2, -(l + l * expend_ratio) / 2]])
    rotate_matrix = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    rel_rot_vertex = transform(rel_vertex, rotate_matrix)
    vertex = rel_rot_vertex + [x, y]  # Global coor sys
    bbox_polygon = Polygon(vertex)
    return bbox_polygon


def get_interior_scene_points(points, bbox, limit, offset=[0.5, 0.5, -0.2]):
    w, l, h, x, y, z, r, cls, diff = bbox
    rel_point_x = points[:, 0] - x
    rel_point_y = points[:, 1] - y
    rel_point_z = points[:, 2] - z
    rot_rel_point_x = rel_point_x * np.cos(-r) - rel_point_y * np.sin(-r)
    rot_rel_point_y = rel_point_x * np.sin(-r) + rel_point_y * np.cos(-r)
    interior_idx = (np.abs(rot_rel_point_x) <= w / 2 + offset[0]) * \
                   (np.abs(rot_rel_point_y) <= l / 2 + offset[1]) * \
                   (np.abs(rel_point_z) <= h / 2 + offset[2])
    if np.sum(interior_idx) > limit:
        return True, points
    else:
        return False, points[interior_idx == 0]


def ground_align(points, bbox, ground, trans_list):
    a, b, c, d = ground
    P2, R0_rect, Tr_velo_to_cam = trans_list
    trans_lidar_to_cam = R0_rect.dot(Tr_velo_to_cam)
    trans_cam_to_lidar = np.linalg.inv(trans_lidar_to_cam)

    x_l, y_l, z_l = bbox[3:6]
    h = bbox[2]
    x_c, y_c, z_c, _ = trans_lidar_to_cam.dot(np.array([x_l, y_l, z_l, 1.])).transpose().tolist()
    y_c0 = (- d - a * x_c - c * z_c) / b
    x_l0, y_l0, z_l0, _ = trans_cam_to_lidar.dot(np.array([x_c, y_c0, z_c, 1.])).transpose().tolist()
    bbox[5] = z_l0 + h / 2.
    points[:, 2] += bbox[5] - z_l

    return np.array(points), np.array(bbox)

def object_points_shift(point, bbox, offset, x_min, y_min, x_max, y_max):
    center_xy = np.array(bbox[3:5])
    point[:, :2] = point[:, :2] - center_xy
    # random_r = np.random.uniform(low=-2*np.pi, high=2*np.pi)
    # # random_r = 0.
    # T = np.array([[np.cos(random_r), -np.sin(random_r)],
    #               [np.sin(random_r), np.cos(random_r)]])
    #
    # point[:, :2] = transform(point[:, :2], T)

    random_xy = np.array([np.random.uniform(low=-offset, high=offset),
                          np.random.uniform(low=-offset, high=offset)])
    offset_xy = center_xy + random_xy
    offset_xy[0] = np.clip(offset_xy[0], x_min, x_max)
    offset_xy[1] = np.clip(offset_xy[1], y_min, y_max)

    point[:, :2] = point[:, :2] + offset_xy
    bbox[3:5] = offset_xy
    # bbox[6] += random_r
    return point, bbox


def get_pasted_point_cloud_waymo(scene_points, scene_bboxes, object_collections, bbox_collections,
                                 instance_num, maximum_interior_points):
    bbox_polygons = []
    output_points = []
    output_bboxes = []

    for i in range(len(scene_bboxes)):
        output_bboxes.append(scene_bboxes[i])
        bbox_polygons.append(get_polygon_from_bbox(deepcopy(scene_bboxes[i])))

    for i in range(instance_num - len(scene_bboxes)):
        found = False
        count = 0
        while not found and count < 10:
            id = np.random.randint(len(object_collections))
            new_points = object_collections[id]
            new_bbox = bbox_collections[id]
            count += 1

            new_bbox = deepcopy(new_bbox)
            new_points = deepcopy(new_points)
            new_points, new_bbox = object_points_shift(new_points, new_bbox, offset=5, x_min=-70, y_min=-70, x_max=70, y_max=70)
            new_bbox_polygon = Polygon(get_polygon_from_bbox(new_bbox, expend_ratio=0.15))
            overlap = False
            intersect = False
            for polygon in bbox_polygons:
                if new_bbox_polygon.intersection(polygon).area > 0:
                    overlap = True
                    break

            if not overlap:
                intersect, scene_points = get_interior_scene_points(points=scene_points,
                                                                    bbox=new_bbox,
                                                                    limit=maximum_interior_points)

            if not overlap and not intersect:
                bbox_polygons.append(new_bbox_polygon)
                output_points.append(new_points)
                output_bboxes.append(new_bbox)
                found = True

    output_points.append(scene_points)
    output_points = np.concatenate(output_points, axis=0)
    if len(output_bboxes) > 0:
        output_bboxes = np.stack(output_bboxes, axis=0)

    return output_points, output_bboxes






def get_pasted_point_cloud(scene_points, scene_bboxes, ground, trans_list, object_collections, bbox_collections,
                           instance_num, maximum_interior_points):
    bbox_polygons = []
    output_points = []
    output_bboxes = []

    for i in range(len(scene_bboxes)):
        bbox_polygons.append(get_polygon_from_bbox(deepcopy(scene_bboxes[i])))
        output_bboxes.append(scene_bboxes[i])

    for i in range(instance_num):
        prob = np.random.rand()

        if prob < 0.2:
            id = np.random.randint(len(object_collections[0]))
            new_points = object_collections[0][id]
            new_bbox = bbox_collections[0][id]
        elif prob < 0.6:
            id = np.random.randint(len(object_collections[1]))
            new_points = object_collections[1][id]
            new_bbox = bbox_collections[1][id]
        else:
            id = np.random.randint(len(object_collections[2]))
            new_points = object_collections[2][id]
            new_bbox = bbox_collections[2][id]

        new_bbox = deepcopy(new_bbox)
        new_points = deepcopy(new_points)
        new_points, new_bbox = object_points_shift(new_points, new_bbox, offset=5, x_min=5, y_min=-35, x_max=65, y_max=35)
        new_points, new_bbox = ground_align(new_points, new_bbox, ground, trans_list)
        new_bbox_polygon = Polygon(get_polygon_from_bbox(new_bbox, expend_ratio=0.15))
        overlap = False
        intersect = False
        for polygon in bbox_polygons:
            if new_bbox_polygon.intersection(polygon).area > 0:
                overlap = True
                break

        if not overlap:
            intersect, scene_points = get_interior_scene_points(points=scene_points,
                                                                bbox=new_bbox,
                                                                limit=maximum_interior_points)

        if not overlap and not intersect:
            bbox_polygons.append(new_bbox_polygon)
            output_points.append(new_points)
            output_bboxes.append(new_bbox)

    output_points.append(scene_points)
    output_points = np.concatenate(output_points, axis=0)
    if len(output_bboxes) > 0:
        output_bboxes = np.stack(output_bboxes, axis=0)

    return output_points, output_bboxes
