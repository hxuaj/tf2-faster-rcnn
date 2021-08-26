from config.config import cfg
import tensorflow as tf
from tensorflow.keras import layers
from model.utils.bbox_transform import delta2bbox, clip_to_boundary


class proposal_layer(layers.Layer):
    """
    Proposal layer
    A tf.keras.layers.Layer receives RPN class and bbox predictions 
    then produces region-of-interests.
    
    Args:
    - is_training: bool, flag to switch network mode between training and testing.
    - pre_nms_top_n: int, upper limit number of top-score boxes before Non Max Suppression
    - post_nms_top_n: int, upper limit number of top-score boxes after Non Max Suppression
    - nms_thresh: float, IoU threshold of Non Max Suppression
    
    """
    
    def __init__(self, is_training):
        super(proposal_layer, self).__init__()
        self.is_training = is_training
        if self.is_training:
            self.pre_nms_top_n = tf.cast(cfg.train_pre_nms_top_n, dtype=tf.int32)
            self.post_nms_top_n = tf.cast(cfg.train_post_nms_top_n, dtype=tf.int32)
        else:
            self.pre_nms_top_n = tf.cast(cfg.test_pre_nms_top_n, dtype=tf.int32)
            self.post_nms_top_n = tf.cast(cfg.test_post_nms_top_n, dtype=tf.int32)
        self.nms_thresh = tf.cast(cfg.nms_thresh, dtype=tf.float32)
    
    @tf.function(experimental_relax_shapes=True)
    def call(self, rpn_cls_prob, rpn_bbox_pred, anchors, img_size):
        """
        Call method for proposal layer.
        
        Inputs:
        - rpn_cls_prob: tf.float, class probabilities after softmax, shape=(n, h, w, num_anchors, 2)
        - rpn_bbox_pred: tf.float, bounding box deltas, shape=(n, h, w, num_anchors, 4)
        - anchors: tf.float, anchor bounding boxes based on feature map size and anchor reference. 
                   shape=(h*w*num_anchors, 4)
        - img_size: tf.int, input scaled image size
        
        Outputs:
        - rois: tf.float, proposed Region of Interests, (x_min, y_min, x_max, y_max), shape=(N, 4)
        
        """
        # Caution: Converting sparse IndexedSlices to a dense Tensor of 
        #          unknown shape may consume a large amount of memory.
        # interested in only lable-object's scores
        scores = rpn_cls_prob[:, :, :, :, 1]
        scores = tf.reshape(scores, (-1,)) # (h * w * num_anchors, )
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred, (-1, 4)) # (h * w * num_anchors, 4)
        
        pre_nms_top_n = tf.minimum(self.pre_nms_top_n, tf.shape(anchors)[0])
        pre_index = tf.math.top_k(scores, k=pre_nms_top_n, sorted=True).indices
        anchors_sorted = tf.gather(anchors, pre_index)
        scores_sorted = tf.gather(scores, pre_index)
        rpn_bbox_pred_sorted = tf.gather(rpn_bbox_pred, pre_index)
        
        # Decode anchors and rpn_bbox_prep into proposals
        proposals = delta2bbox(anchors_sorted, rpn_bbox_pred_sorted)
        proposals = clip_to_boundary(proposals, img_size)
        
        # Non-max suppression
        # tf.image.non_max_suppression works on tf.float32, returns tf.int32
        # Althought tf.image.non_max_suppression takes bboxes as (y_min, x_min, y_max, x_max),
        # it's the same to input (x_min, y_min, x_max, y_max), since compare IoU.
        
        proposals_trans = tf.stack([proposals[:, 1], proposals[:, 0], proposals[:, 3], proposals[:, 2]], axis=1)
        indices = tf.image.non_max_suppression(proposals_trans, scores_sorted,
                                               max_output_size=self.post_nms_top_n,
                                               iou_threshold=self.nms_thresh)
        
        # rois: (x_min, y_min, x_max, y_max), shape=(len(indices), 4)
        rois = tf.gather(proposals, indices)
        # roi_scores = tf.gather(scores_sorted, indices)
        
        return rois
