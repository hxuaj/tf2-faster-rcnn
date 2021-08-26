import tensorflow as tf
import numpy as np


@tf.function(experimental_relax_shapes=True)
def delta2bbox(anchors, deltas):
    """
    Bounding box regression decoder
    anchor + deltas -> prediction_box
    
    Input:
    - anchors: tf.float, shape=(h * w * num_anchors, 4), (x_min, y_min, x_max, y_max)
    - deltas: rpn_bbox_pred, tf.float, shape=(h * w * num_anchors, 4), (dx, dy, dw, dh)
    
    Output:
    - (x_min, y_min, x_max, y_max), tf.float, shape=(h * w * num_anchors, 4)
    """
    
    assert anchors.dtype == deltas.dtype, "The data types of anchors({}) and deltas({}) are not match.".format(anchors.dtype,
                                                                                                               deltas.dtype)
    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    widths = anchors[:, 2] - anchors[:, 0] + 1.0
    heights = anchors[:, 3] - anchors[:, 1] + 1.0
    center_x = anchors[:, 0] + widths * 0.5
    center_y = anchors[:, 1] + heights * 0.5
    
    pred_center_x = dx * widths + center_x
    pred_center_y = dy * heights + center_y
    pred_w = tf.exp(dw) * widths
    pred_h = tf.exp(dh) * heights
    
    x_min = pred_center_x - pred_w * 0.5
    y_min = pred_center_y - pred_h * 0.5
    x_max = pred_center_x + pred_w * 0.5
    y_max = pred_center_y + pred_h * 0.5

    return tf.stack([x_min, y_min, x_max, y_max], axis=1)

def clip_to_boundary(bbox, img_size):
    """clip the proposals to the boundary of the scaled image"""
    img_size = tf.stop_gradient(tf.cast(img_size, dtype=bbox.dtype))

    x_min = tf.clip_by_value(bbox[:, 0], clip_value_min=0, clip_value_max=img_size[1] - 1)
    y_min = tf.clip_by_value(bbox[:, 1], clip_value_min=0, clip_value_max=img_size[0] - 1)
    x_max = tf.clip_by_value(bbox[:, 2], clip_value_min=0, clip_value_max=img_size[1] - 1)
    y_max = tf.clip_by_value(bbox[:, 3], clip_value_min=0, clip_value_max=img_size[0] - 1)
    
    return tf.stack([x_min, y_min, x_max, y_max], axis=1)

def bbox_iou(box1, box2):
    """
    Inputs:
    - box1: array, (N, 4)
    - box2: array, (K, 4)
    Outputs:
    - ious: array, (N, K)
    """
    box1_area = np.prod(box1[:, 2:] - box1[:, :2], axis=1)
    box2_area = np.prod(box2[:, 2:] - box2[:, :2], axis=1)
    # overlaps top left, shape=(N, K, 2)
    tl = np.maximum(box1[:, None, :2], box2[None, :, :2])
    # overlaps bottom right, shape(N, K, 2)
    br = np.minimum(box1[:, None, 2:], box2[None, :, 2:])
    # exclude the set of box1 and box2 without overlapping
    mask = np.all(br > tl, axis=2, keepdims=False)
    # overlaps area, shape=(N, K)
    iou_area = np.prod(br - tl, axis=2) * mask
    
    # shape=(N, K)
    ious = iou_area / (box1_area[:, None] + box2_area[None, :] - iou_area)
    
    return ious

def bbox2delta(rois, gt_boxes):
    """
    Bounding box regression encoder
    prediction_box + gt_boxes -> gt_delta
    
    Inputs:
    - rois: sampled rois, array, shape=(S, 4), (x_min, y_min, x_max, y_max)
    - gt_boxes: ground truth boxes correspond to sampled rois, shape=(S, 4), (x_min, y_min, x_max, y_max)
    
    Outputs:
    - gt_delta: sampled rois' ground truth regression delta, shape=(S, 4) , (dx, dy, dw, dh)
    """
    
    w_a = rois[:, 2] - rois[:, 0] + 1.0
    h_a = rois[:, 3] - rois[:, 1] + 1.0
    x_a = rois[:, 0] + w_a * 0.5
    y_a = rois[:, 1] + h_a * 0.5
    
    w_gt = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    h_gt = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    x_gt = gt_boxes[:, 0] + w_gt * 0.5
    y_gt = gt_boxes[:, 1] + h_gt * 0.5
    
    # clamp the the smallest value to be eps rather than zero
    eps = np.finfo(rois.dtype).eps
    w_a = np.maximum(w_a, eps)
    h_a = np.maximum(h_a, eps)
    
    dx = (x_gt - x_a) / w_a
    dy = (y_gt - y_a) / h_a
    dw = np.log(w_gt / w_a)
    dh = np.log(h_gt / h_a)
    
    return np.vstack((dx, dy, dw, dh)).T

def delete_extra(keep):
    """
    Due to the tf2.1 bug: https://github.com/tensorflow/tensorflow/issues/29628
    Delete extra inds of tf.image.non_max_suppression output
    
    """
    n = keep.shape[0]
    if n <= 1:
        return keep
    else:
        for i in range(n - 1):
            if keep[i] == keep[i+1]:
                return keep[:i+1]
        return keep

def transfer_xy(bbox):
    """self-defined transfer bboxes (x1, y1, x2, y2) to (y1, x1, y2, x2)"""
    b = bbox.numpy()
    out = np.concatenate((b[:, 1::4], b[:, 0::4], b[:, 3::4], b[:, 2::4]), axis=1)
    return tf.convert_to_tensor(out, dtype=tf.float32)
