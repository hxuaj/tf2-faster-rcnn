import numpy as np
from config.config import cfg
from model.utils.bbox_transform import bbox_iou, bbox2delta


def anchor_to_target(h, w, num_anchors, gt_boxes, anchors, img_size):
    """
    Assign ground truth bounding boxes to anchors and sample anchors
    for training rpn.
    
    Input:
    - (h, w): tf.int, feature map shape=(n, h, w, c)
    - num_anchors, the number of anchors in one anchor point
    - gt_boxes: tf.int, (x_min, y_min, x_max, y_max), shape=(K, 4)
    - anchors: tf.float, shape=(A, 4)
    - img_size: tf.int, the size of input rescaled image,shape=(img_h, img_w)
    
    Outputs:
    - rpn_labels: np.int, labels of anchors, (positive=1, negative=0, ignore=-1), shape=(n, h, w, num_anchors, 1)
    - rpn_bbox_targets: np.float, shape=(n, h, w, num_anchors, 4)
    """
    
    img_h, img_w = img_size.numpy()
    gt_boxes = gt_boxes.numpy()
    anchors = anchors.numpy()
    A = anchors.shape[0] # ~16000-20000
    
    # keep anchors which are completely inside of the image
    inds_inside = np.where(
        (anchors[:, 0] >= 0) & 
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] < img_w) &
        (anchors[:, 3] < img_h)
    )[0]
    anchors_inside = anchors[inds_inside, :]
    N = len(inds_inside) # ~5000

    # calculate the ious of anchors and gt_boxes
    ious = bbox_iou(anchors_inside, gt_boxes) # (N, K)
    argmax_ious = np.argmax(ious, axis=1) # (N, ) in range(0, K) each anchor's highest iou
    max_ious = ious[np.arange(N), argmax_ious] # N
    gt_argmax_ious = np.argmax(ious, axis=0) # (K, ) in range(0, N)
    gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])] # K
    gt_argmax_ious = np.where(ious == gt_max_ious)[0]
    
    # generate labels
    labels = np.empty((N,), dtype=np.int32)
    labels.fill(-1)

    # assign neg labels before pos, so the pos labels can clobber them
    labels[max_ious < cfg.neg_thresh] = 0
    labels[gt_argmax_ious] = 1 # each gt's highest iou(maybe lower than threshold)
    labels[max_ious >= cfg.pos_thresh] = 1 # ious larger than threshold

    # subsample positive labels if we have too many
    num_pos = int(cfg.pos_sample_ratio * cfg.num_sample_rpn)
    pos_inds = np.where(labels == 1)[0]
    
    if len(pos_inds) > num_pos:
        disable_index = np.random.choice(
            pos_inds, size=(len(pos_inds) - num_pos), replace=False)
        labels[disable_index] = -1
    # subsample negative labels if we have too many
    num_neg = cfg.num_sample_rpn - np.sum(labels == 1)
    neg_inds = np.where(labels == 0)[0]
    if len(neg_inds) > num_neg:
        disable_index = np.random.choice(
            neg_inds, size=(len(neg_inds) - num_neg), replace=False)
        labels[disable_index] = -1
    
    # calculate rpn bounding box regression targets
    rpn_bbox_targets = bbox2delta(anchors_inside, gt_boxes[argmax_ious, :]) # (N, 4)
    
    # unmap to original shape of anchors (A, 4)
    rpn_labels = _unmap(labels, A, inds_inside, fill=-1)
    rpn_bbox_targets = _unmap(rpn_bbox_targets, A, inds_inside, fill=0)
    
    # reshape
    rpn_labels = np.reshape(rpn_labels, (1, h, w, num_anchors, 1))
    rpn_bbox_targets = np.reshape(rpn_bbox_targets, (1, h, w, num_anchors, 4))
    
    return rpn_labels, rpn_bbox_targets


def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret
