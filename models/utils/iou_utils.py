import tensorflow as tf
import horovod.tensorflow as hvd

# Ground Truth Shape: [npoint, 7 (w, l, h, x, y, z, r)]
# Prediction Shape: [npoint, 7 (w, l, h, x, y, z, r)]


'''
         x (w)
         ^
0        |        1
 |---------------|
 |       *       |  ---> y (l)
 |---------------|
3                 2
'''

eps = tf.constant(1e-6)



# def roi_logits_to_attrs_tf(base_coors, input_logits, anchor_size):
#     anchor_diag = tf.sqrt(tf.pow(anchor_size[0], 2.) + tf.pow(anchor_size[1], 2.))
#     w = tf.clip_by_value(tf.exp(input_logits[:, 0]) * anchor_size[0], 0., 1e5)
#     l = tf.clip_by_value(tf.exp(input_logits[:, 1]) * anchor_size[1], 0., 1e5)
#     h = tf.clip_by_value(tf.exp(input_logits[:, 2]) * anchor_size[2], 0., 1e5)
#     x = tf.clip_by_value(input_logits[:, 3] * anchor_diag + base_coors[:, 0], -1e5, 1e5)
#     y = tf.clip_by_value(input_logits[:, 4] * anchor_diag + base_coors[:, 1], -1e5, 1e5)
#     z = tf.clip_by_value(input_logits[:, 5] * anchor_size[2] + base_coors[:, 2], -1e5, 1e5)
#     r = input_logits[:, 6] * 3.1415927
#     return tf.stack([w, l, h, x, y, z, r], axis=-1)

def roi_logits_to_attrs_tf(base_coors, input_logits, anchor_size):
    anchor_diag = tf.sqrt(tf.pow(anchor_size[0], 2.) + tf.pow(anchor_size[1], 2.))
    w = tf.clip_by_value(tf.exp(input_logits[:, 0]) * anchor_size[0], 0., 1e7)
    l = tf.clip_by_value(tf.exp(input_logits[:, 1]) * anchor_size[1], 0., 1e7)
    h = tf.clip_by_value(tf.exp(input_logits[:, 2]) * anchor_size[2], 0., 1e7)
    x = tf.clip_by_value(input_logits[:, 3] * anchor_diag + base_coors[:, 0], -1e7, 1e7)
    y = tf.clip_by_value(input_logits[:, 4] * anchor_diag + base_coors[:, 1], -1e7, 1e7)
    z = tf.clip_by_value(input_logits[:, 5] * anchor_size[2] + base_coors[:, 2], -1e7, 1e7)
    r = tf.clip_by_value(input_logits[:, 6] * 3.1415927, -1e7, 1e7)
    # r = input_logits[:, 6]
    return tf.stack([w, l, h, x, y, z, r], axis=-1)


def roi_attrs_to_logits(base_coors, input_attrs, anchor_size):
    anchor_diag = tf.sqrt(tf.pow(anchor_size[0], 2.) + tf.pow(anchor_size[1], 2.))
    logits_w = tf.log(input_attrs[:, 0] / anchor_size[0])
    logits_l = tf.log(input_attrs[:, 1] / anchor_size[1])
    logits_h = tf.log(input_attrs[:, 2] / anchor_size[2])
    logits_x = (input_attrs[:, 3] - base_coors[:, 0]) / anchor_diag
    logits_y = (input_attrs[:, 4] - base_coors[:, 1]) / anchor_diag
    logits_z = (input_attrs[:, 5] - base_coors[:, 2]) / anchor_size[2]
    logits_r = input_attrs[:, 6] / 3.1415927
    return tf.stack([logits_w, logits_l, logits_h, logits_x, logits_y, logits_z, logits_r], axis=-1)


def bbox_logits_to_attrs_tf(input_roi_attrs, input_logits):
    roi_diag = tf.sqrt(tf.pow(input_roi_attrs[:, 0], 2.) + tf.pow(input_roi_attrs[:, 1], 2.))
    w = tf.clip_by_value(tf.exp(input_logits[:, 0]) * input_roi_attrs[:, 0], 0., 1e7)
    l = tf.clip_by_value(tf.exp(input_logits[:, 1]) * input_roi_attrs[:, 1], 0., 1e7)
    h = tf.clip_by_value(tf.exp(input_logits[:, 2]) * input_roi_attrs[:, 2], 0., 1e7)
    x = tf.clip_by_value(input_logits[:, 3] * roi_diag + input_roi_attrs[:, 3], -1e7, 1e7)
    y = tf.clip_by_value(input_logits[:, 4] * roi_diag + input_roi_attrs[:, 4], -1e7, 1e7)
    z = tf.clip_by_value(input_logits[:, 5] * input_roi_attrs[:, 2] + input_roi_attrs[:, 5], -1e7, 1e7)
    r = tf.clip_by_value(input_logits[:, 6] * 3.1415927 + input_roi_attrs[:, 6], -1e7, 1e7)
    # r = input_logits[:, 6] + input_roi_attrs[:, 6]
    return tf.stack([w, l, h, x, y, z, r], axis=-1)


def get_rotate_matrix(r):
    rotate_matrix = tf.stack([tf.cos(r), -tf.sin(r), tf.sin(r), tf.cos(r)], axis=-1)  # [n, 4]
    rotate_matrix = tf.reshape(rotate_matrix, shape=[-1, 2, 2])  # [n, 2, 2]
    return rotate_matrix


def get_2d_vertex_points(gt_attrs, pred_attrs):
    gt_w = gt_attrs[:, 0]  # [n]
    gt_l = gt_attrs[:, 1]  # [n]
    gt_x = gt_attrs[:, 3]  # [n]
    gt_y = gt_attrs[:, 4]  # [n]
    gt_r = gt_attrs[:, 6]  # [n]

    gt_v0 = tf.stack([gt_w / 2, -gt_l / 2], axis=-1)  # [n, 2]
    gt_v1 = tf.stack([gt_w / 2, gt_l / 2], axis=-1)  # [n, 2]
    gt_v2 = tf.stack([-gt_w / 2, gt_l / 2], axis=-1)  # [n, 2]
    gt_v3 = tf.stack([-gt_w / 2, -gt_l / 2], axis=-1)  # [n, 2]
    gt_v = tf.stack([gt_v0, gt_v1, gt_v2, gt_v3], axis=1)  # [n, 4, 2]

    pred_w = pred_attrs[:, 0]  # [n]
    pred_l = pred_attrs[:, 1]  # [n]
    pred_x = pred_attrs[:, 3]  # [n]
    pred_y = pred_attrs[:, 4]  # [n]
    pred_r = pred_attrs[:, 6]  # [n]

    rel_x = pred_x - gt_x  # [n]
    rel_y = pred_y - gt_y  # [n]
    rel_r = pred_r - gt_r  # [n]
    rel_xy = tf.expand_dims(tf.stack([rel_x, rel_y], axis=-1), axis=1)  # [n, 1, 2]

    pred_v0 = tf.stack([pred_w / 2, -pred_l / 2], axis=-1)  # [n, 2]
    pred_v1 = tf.stack([pred_w / 2, pred_l / 2], axis=-1)  # [n, 2]
    pred_v2 = tf.stack([-pred_w / 2, pred_l / 2], axis=-1)  # [n, 2]
    pred_v3 = tf.stack([-pred_w / 2, -pred_l / 2], axis=-1)  # [n, 2]
    pred_v = tf.stack([pred_v0, pred_v1, pred_v2, pred_v3], axis=1)  # [n, 4, 2]

    rot_pred_v = tf.transpose(tf.matmul(a=get_rotate_matrix(rel_r), b=pred_v, transpose_b=True),
                              perm=[0, 2, 1])  # [n, 4, 2]
    rot_rel_xy = tf.transpose(tf.matmul(a=get_rotate_matrix(-gt_r), b=rel_xy, transpose_b=True),
                              perm=[0, 2, 1])  # [n, 1, 2]
    rel_rot_pred_v = rot_pred_v + rot_rel_xy  # [n, 4, 2]

    rot_gt_v = tf.transpose(tf.matmul(a=get_rotate_matrix(-rel_r), b=gt_v, transpose_b=True),
                            perm=[0, 2, 1])  # [n, 4, 2]
    rot_rel_xy = tf.transpose(tf.matmul(a=get_rotate_matrix(-pred_r), b=-rel_xy, transpose_b=True),
                              perm=[0, 2, 1])  # [n, 1, 2]
    rel_rot_gt_v = rot_gt_v + rot_rel_xy  # [n, 4, 2]

    # [n, 2, 2] @ [n, 2, 4] = [n, 2, 4] -> [n, 4, 2]

    return gt_v, rel_rot_pred_v, rel_rot_gt_v, rel_xy, rel_r


def get_2d_intersection_points(gt_attrs, rel_rot_pred_v):
    gt_w = gt_attrs[:, 0]  # [n]
    gt_l = gt_attrs[:, 1]  # [n]
    output_points = []
    for i in [-1, 0, 1, 2]:
        v0_x = rel_rot_pred_v[:, i, 0]  # [n]
        v0_y = rel_rot_pred_v[:, i, 1]  # [n]
        v1_x = rel_rot_pred_v[:, i + 1, 0]  # [n]
        v1_y = rel_rot_pred_v[:, i + 1, 1]  # [n]

        kx = tf.math.divide_no_nan(v1_y - v0_y, v1_x - v0_x)
        bx = tf.math.divide_no_nan(v0_y * v1_x - v1_y * v0_x, v1_x - v0_x)
        ky = tf.math.divide_no_nan(v1_x - v0_x, v1_y - v0_y)
        by = tf.math.divide_no_nan(v1_y * v0_x - v0_y * v1_x, v1_y - v0_y)

        # kx = (v1_y - v0_y) / (v1_x - v0_x + eps) # [n]
        # bx = (v0_y * v1_x - v1_y * v0_x) / (v1_x - v0_x + eps) # [n]
        # ky = (v1_x - v0_x) / (v1_y - v0_y + eps) # [n]
        # by = (v1_y * v0_x - v0_y * v1_x) / (v1_y - v0_y + eps) # [n]

        p0 = tf.stack([gt_w / 2, kx * gt_w / 2 + bx], axis=-1)  # [n, 2]
        p1 = tf.stack([-gt_w / 2, -kx * gt_w / 2 + bx], axis=-1)  # [n, 2]
        p2 = tf.stack([ky * gt_l / 2 + by, gt_l / 2], axis=-1)  # [n, 2]
        p3 = tf.stack([-ky * gt_l / 2 + by, -gt_l / 2], axis=-1)  # [n, 2]
        p = tf.stack([p0, p1, p2, p3], axis=1)  # [n, 4, 2]
        output_points.append(p)
    output_points = tf.concat(output_points, axis=1)  # [n, 16, 2]
    return output_points


def get_interior_vertex_points_mask(target_attrs, input_points):
    target_w = tf.expand_dims(target_attrs[:, 0], axis=1)  # [n, 1, 16]
    target_l = tf.expand_dims(target_attrs[:, 1], axis=1)  # [n, 1, 16]
    target_x = target_w / 2  # [n, 4]
    target_y = target_l / 2  # [n, 4]
    x_mask = tf.cast(tf.less_equal(tf.abs(input_points[:, :, 0]), target_x), dtype=tf.float32)  # [n, 4]
    y_mask = tf.cast(tf.less_equal(tf.abs(input_points[:, :, 1]), target_y), dtype=tf.float32)  # [n, 4]
    return x_mask * y_mask  # [n, 4]


#
#
def get_intersection_points_mask(target_attrs, input_points, rel_xy=None, rel_r=None):
    if rel_xy is not None and rel_r is not None:
        pred_r = target_attrs[:, 6]  # [n]
        rot_input_points = tf.transpose(tf.matmul(a=get_rotate_matrix(-rel_r), b=input_points, transpose_b=True),
                                        perm=[0, 2, 1])  # [n, 16, 2]
        rot_rel_xy = tf.transpose(tf.matmul(a=get_rotate_matrix(-pred_r), b=-rel_xy, transpose_b=True),
                                  perm=[0, 2, 1])  # [n, 1, 2]
        rel_rot_input_points = rot_input_points + rot_rel_xy
    else:
        rel_rot_input_points = input_points
    target_w = tf.expand_dims(target_attrs[:, 0], axis=1)  # [n, 1, 16]
    target_l = tf.expand_dims(target_attrs[:, 1], axis=1)  # [n, 1, 16]
    target_x = target_w / 2 + 1e-3  # [n, 4]
    target_y = target_l / 2 + 1e-3  # [n, 4]
    # target_x = 1000  # [n, 4]
    # target_y = 1000  # [n, 4]
    max_x_mask = tf.cast(tf.less_equal(tf.abs(rel_rot_input_points[:, :, 0]), target_x), dtype=tf.float32)  # [n, 4]
    max_y_mask = tf.cast(tf.less_equal(tf.abs(rel_rot_input_points[:, :, 1]), target_y), dtype=tf.float32)  # [n, 4]
    return max_x_mask * max_y_mask  # [n, 4]


def clockwise_sorting(input_points, masks):
    coors_masks = tf.stack([masks, masks], axis=-1)  # [n, 24, 2]
    masked_points = input_points * coors_masks
    centers = tf.math.divide_no_nan(tf.reduce_sum(masked_points, axis=1, keepdims=True),
                                    (tf.reduce_sum(coors_masks, axis=1, keepdims=True)))  # [n, 1, 2]
    rel_vectors = input_points - centers  # [n, 24, 2]
    base_vector = rel_vectors[:, :1, :]  # [n, 1, 2]
    # https://stackoverflow.com/questions/14066933/direct-way-of-computing-clockwise-angle-between-2-vectors/16544330#16544330
    dot = base_vector[:, :, 0] * rel_vectors[:, :, 0] + base_vector[:, :, 1] * rel_vectors[:, :, 1]  # [n, 24]
    det = base_vector[:, :, 0] * rel_vectors[:, :, 1] - base_vector[:, :, 1] * rel_vectors[:, :, 0]  # [n, 24]
    angles = tf.math.atan2(det + eps, dot + eps)  # [n, 24] -pi~pi
    angles_masks = (0.5 - (masks - 0.5)) * 1000.  # [n, 24]
    masked_angles = angles + angles_masks  # [n, 24]
    # _, sort_idx = tf.nn.top_k(-masked_angles, k=input_points.get_shape().as_list()[1], sorted=True)  # [n, 24]
    _, sort_idx = tf.nn.top_k(-masked_angles, k=tf.shape(input_points)[1], sorted=True)  # [n, 24]

    batch_id = tf.expand_dims(tf.range(start=0, limit=tf.shape(input_points)[0], dtype=tf.int32), axis=1)
    # batch_ids = tf.stack([batch_id] * input_points.get_shape().as_list()[1], axis=1)
    batch_ids = tf.tile(batch_id, [1, tf.shape(input_points)[1]])
    sort_idx = tf.stack([batch_ids, sort_idx], axis=-1)  # [n, 24, 2]

    sorted_points = tf.gather_nd(input_points, sort_idx)
    sorted_masks = tf.gather_nd(masks, sort_idx)

    return sorted_points, sorted_masks


def shoelace_intersection_area(sorted_points, sorted_masks):
    # https://en.wikipedia.org/wiki/Shoelace_formula
    sorted_points = sorted_points * tf.stack([sorted_masks, sorted_masks], axis=-1)  # [n, 24, 2]
    last_vertex_id = tf.cast(tf.reduce_sum(sorted_masks, axis=1) - 1,
                             dtype=tf.int32)  # [n] coors where idx=-1 will be convert to [0., 0.], so it's safe.
    last_vertex_id = tf.stack([tf.range(start=0, limit=tf.shape(sorted_points)[0], dtype=tf.int32), last_vertex_id],
                              axis=-1)  # [n, 2]
    last_vertex_to_duplicate = tf.expand_dims(tf.gather_nd(sorted_points, last_vertex_id), axis=1)  # [n, 1, 2]
    padded_sorted_points = tf.concat([last_vertex_to_duplicate, sorted_points], axis=1)  # [n, 24+1, 2]
    x_i = padded_sorted_points[:, :-1, 0]  # [n, 24]
    x_i_plus_1 = padded_sorted_points[:, 1:, 0]  # [n, 24]
    y_i = padded_sorted_points[:, :-1, 1]  # [n, 24]
    y_i_plus_1 = padded_sorted_points[:, 1:, 1]  # [n, 24]
    area = 0.5 * tf.reduce_sum(x_i * y_i_plus_1 - x_i_plus_1 * y_i, axis=-1)  # [n]
    return area


def get_intersection_height(gt_attrs, pred_attrs):
    gt_h = gt_attrs[:, 2]
    gt_z = gt_attrs[:, 5]
    pred_h = pred_attrs[:, 2]
    pred_z = pred_attrs[:, 5]
    gt_low = gt_z - 0.5 * gt_h
    gt_high = gt_z + 0.5 * gt_h
    pred_low = pred_z - 0.5 * pred_h
    pred_high = pred_z + 0.5 * pred_h
    top = tf.minimum(gt_high, pred_high)
    bottom = tf.maximum(gt_low, pred_low)
    intersection_height = tf.nn.relu(top - bottom)
    return intersection_height


def get_3d_iou_from_area(gt_attrs, pred_attrs, intersection_2d_area, intersection_height, clip):
    intersection_volume = intersection_2d_area * intersection_height
    gt_volume = gt_attrs[:, 0] * gt_attrs[:, 1] * gt_attrs[:, 2]
    pred_volume = pred_attrs[:, 0] * pred_attrs[:, 1] * pred_attrs[:, 2]
    iou = tf.math.divide_no_nan(intersection_volume, gt_volume + pred_volume - intersection_volume)
    # tf.summary.scalar('iou_nan_sum',
    #                   hvd.allreduce(tf.reduce_sum(tf.cast(tf.is_nan(iou), dtype=tf.float32)), average=False))
    if clip:
        iou = tf.where(tf.is_nan(iou), tf.zeros_like(iou), iou)
    return iou

def get_bev_iou_from_area(gt_attrs, pred_attrs, intersection_2d_area, clip):
    gt_area = gt_attrs[:, 0] * gt_attrs[:, 1]
    pred_area = pred_attrs[:, 0] * pred_attrs[:, 1]
    iou = tf.math.divide_no_nan(intersection_2d_area, gt_area + pred_area - intersection_2d_area)
    # tf.summary.scalar('iou_nan_sum',
    #                   hvd.allreduce(tf.reduce_sum(tf.cast(tf.is_nan(iou), dtype=tf.float32)), average=False))
    if clip:
        iou = tf.where(tf.is_nan(iou), tf.zeros_like(iou), iou)
    return iou


def cal_3d_iou(gt_attrs, pred_attrs, clip=False):
    gt_v, rel_rot_pred_v, rel_rot_gt_v, rel_xy, rel_r = get_2d_vertex_points(gt_attrs, pred_attrs)
    intersection_points = get_2d_intersection_points(gt_attrs=gt_attrs, rel_rot_pred_v=rel_rot_pred_v)
    gt_vertex_points_inside_pred = get_interior_vertex_points_mask(target_attrs=pred_attrs, input_points=rel_rot_gt_v)
    pred_vertex_points_inside_gt = get_interior_vertex_points_mask(target_attrs=gt_attrs, input_points=rel_rot_pred_v)
    pred_intersect_with_gt = get_intersection_points_mask(target_attrs=gt_attrs, input_points=intersection_points)
    intersection_points_inside_pred = get_intersection_points_mask(target_attrs=pred_attrs,
                                                                   input_points=intersection_points, rel_xy=rel_xy,
                                                                   rel_r=rel_r)
    total_points = tf.concat([gt_v, rel_rot_pred_v, intersection_points], axis=1)
    total_masks = tf.concat([gt_vertex_points_inside_pred, pred_vertex_points_inside_gt,
                             pred_intersect_with_gt * intersection_points_inside_pred], axis=1)
    sorted_points, sorted_masks = clockwise_sorting(input_points=total_points, masks=total_masks)

    intersection_2d_area = shoelace_intersection_area(sorted_points, sorted_masks)
    intersection_height = get_intersection_height(gt_attrs, pred_attrs)
    ious = get_3d_iou_from_area(gt_attrs, pred_attrs, intersection_2d_area, intersection_height, clip)

    return ious

def cal_bev_iou(gt_attrs, pred_attrs, clip=False):
    gt_v, rel_rot_pred_v, rel_rot_gt_v, rel_xy, rel_r = get_2d_vertex_points(gt_attrs, pred_attrs)
    intersection_points = get_2d_intersection_points(gt_attrs=gt_attrs, rel_rot_pred_v=rel_rot_pred_v)
    gt_vertex_points_inside_pred = get_interior_vertex_points_mask(target_attrs=pred_attrs, input_points=rel_rot_gt_v)
    pred_vertex_points_inside_gt = get_interior_vertex_points_mask(target_attrs=gt_attrs, input_points=rel_rot_pred_v)
    pred_intersect_with_gt = get_intersection_points_mask(target_attrs=gt_attrs, input_points=intersection_points)
    intersection_points_inside_pred = get_intersection_points_mask(target_attrs=pred_attrs,
                                                                   input_points=intersection_points, rel_xy=rel_xy,
                                                                   rel_r=rel_r)
    total_points = tf.concat([gt_v, rel_rot_pred_v, intersection_points], axis=1)
    total_masks = tf.concat([gt_vertex_points_inside_pred, pred_vertex_points_inside_gt,
                             pred_intersect_with_gt * intersection_points_inside_pred], axis=1)
    sorted_points, sorted_masks = clockwise_sorting(input_points=total_points, masks=total_masks)

    intersection_2d_area = shoelace_intersection_area(sorted_points, sorted_masks)
    ious = get_bev_iou_from_area(gt_attrs, pred_attrs, intersection_2d_area, clip)

    return ious


def cal_3d_iou_debug(gt_attrs, pred_attrs, clip=False):
    gt_v, rel_rot_pred_v, rel_rot_gt_v, rel_xy, rel_r = get_2d_vertex_points(gt_attrs, pred_attrs)
    intersection_points = get_2d_intersection_points(gt_attrs=gt_attrs, rel_rot_pred_v=rel_rot_pred_v)
    gt_vertex_points_inside_pred = get_interior_vertex_points_mask(target_attrs=pred_attrs, input_points=rel_rot_gt_v)
    pred_vertex_points_inside_gt = get_interior_vertex_points_mask(target_attrs=gt_attrs, input_points=rel_rot_pred_v)
    pred_intersect_with_gt = get_intersection_points_mask(target_attrs=gt_attrs, input_points=intersection_points)
    intersection_points_inside_pred = get_intersection_points_mask(target_attrs=pred_attrs,
                                                                   input_points=intersection_points, rel_xy=rel_xy,
                                                                   rel_r=rel_r)
    total_points = tf.concat([gt_v, rel_rot_pred_v, intersection_points], axis=1)
    total_masks = tf.concat([gt_vertex_points_inside_pred, pred_vertex_points_inside_gt,
                             pred_intersect_with_gt * intersection_points_inside_pred], axis=1)
    sorted_points, sorted_masks = clockwise_sorting(input_points=total_points, masks=total_masks)

    intersection_2d_area = shoelace_intersection_area(sorted_points, sorted_masks)
    intersection_height = get_intersection_height(gt_attrs, pred_attrs)
    ious = get_3d_iou_from_area(gt_attrs, pred_attrs, intersection_2d_area, intersection_height, clip)

    return ious, intersection_2d_area
