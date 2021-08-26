import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Softmax
from config.config import cfg
from model.utils.generate_anchors import generate_anchor_ref, generate_anchors
from model.layer.proposal_layer import proposal_layer
from model.layer.proposal_to_target import proposal_to_target
from model.layer.anchor_to_target import anchor_to_target


class RPN(tf.keras.Model):
    """
    Region Proposal Network
    The network takes feature maps extracted from base net and predicts on class and
    bbox delta to make proposals.
    
    Args:
    - anchor_scales: list, scale up anchor width and height
    - anchor_ratios: list, anchors width to height ratio
    - feat_stride: int, feature map is feat_stride times smaller than origin image(downsampling)
    - is_training: bool, Flag to switch network mode between training and testing.
    - weight_decay: float, weight decay param for l2 regularization
    - initializer: kernal initializer
    - regularizer: tensorflow l2 regularizer
    - num_anchors: tf.int32, number of anchors per anchor point
    - anchors_ref: tf.float32, anchor references at upper-left cornner of image
    
    """
    
    def __init__(self, is_training, anchor_scales=[8, 16, 32], anchor_ratios=[0.5, 1, 2], 
                 feat_stride=cfg.feat_stride):
        super(RPN, self).__init__()
        # Args:
        self.is_training = is_training
        self.weight_decay = cfg.weight_decay
        self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        self.regularizer = tf.keras.regularizers.l2(self.weight_decay)
        self.num_anchors = tf.cast(len(anchor_scales) * len(anchor_ratios), dtype=tf.int32)
        self.feat_stride = tf.constant(feat_stride, dtype=tf.int32)
        self.anchors_ref = generate_anchor_ref(scales=anchor_scales, ratios=anchor_ratios)
        
        # Layers:
        self.rpn_conv = Conv2D(512, (3, 3), kernel_regularizer=self.regularizer, activation='relu', 
                               kernel_initializer=self.initializer, padding='same', name='rpn_conv')
        self.score_conv = Conv2D(self.num_anchors * 2, (1, 1), kernel_regularizer=self.regularizer,
                                 activation=None, kernel_initializer=self.initializer, name='score_conv')
        self.bbox_conv = Conv2D(self.num_anchors * 4, (1, 1), kernel_regularizer=self.regularizer,
                                activation=None, kernel_initializer=self.initializer, name='bbox_conv')
        self.softmax = Softmax(axis=-1)
        self.proposal_layer = proposal_layer(self.is_training)
        
        # for calculating losses
        self.rpn_cls_score = 0
        self.rpn_bbox_pred = 0
        
    def call(self, feature_map, img_size, gt_boxes=None, gt_cls=None):
        """
        Call method for RPN model.
        
        Notation:
        - K: number of ground truth objects in image
        - N: number of proposed rois(testing)
        - S: number of sampled rois(training)
        
        Inputs:
        - feature_map: tf.float, features extracted from base net, shape=(n, h, w, c)
        - img_size: tf.int, input scaled image size
        - gt_boxes: tf.int, ground truth bounding box, (x_min, y_min, x_max, y_max), shape=(K, 4)
        - gt_cls: tf.int, ground truth class label, shape=(K, 1)
        
        Outputs:
        - rois: tf.float, proposed Region of Interests, (x_min, y_min, x_max, y_max), shape=(N/S, 4)
        - roi_bbox_targets: tf.float, bbox regression targets, shape=(S, 4)
        - roi_gt_labels: tf.int, sampled roi labels, shape=(S, 1)
        - rpn_labels: tf.int, labels of anchors, shape=(n, h, w, num_anchors, 1)
        - rpn_bbox_targets: tf.float, shape=(n, h, w, num_anchors, 4)
        """
        
        (anchors, rpn_cls_prob, self.rpn_cls_score,
         self.rpn_bbox_pred, h, w) = self.rpn_body(feature_map)
        
        if self.is_training:
            # anchors_to_target
            rpn_labels, rpn_bbox_targets = anchor_to_target(h, w, self.num_anchors, 
                                                            gt_boxes, anchors, img_size)
            
            # rpn_labels, rpn_bbox_targets = tf.py_function(func=anchor_to_target,
            #                                               inp=[h, w, self.num_anchors, 
            #                                                    gt_boxes, anchors, img_size],
            #                                               Tout=[tf.float32, tf.float32])
            rpn_labels = tf.stop_gradient(rpn_labels)
            rpn_bbox_targets = tf.stop_gradient(rpn_bbox_targets)           
            
            rois = self.proposal_layer(rpn_cls_prob, self.rpn_bbox_pred,
                                       anchors, img_size)
            
            # proposal_to_target
            rois, roi_bbox_targets, roi_gt_labels = proposal_to_target(rois, gt_boxes, gt_cls)
            
            # rois, roi_bbox_targets, roi_gt_labels = tf.py_function(func=proposal_to_target,
            #                                                        inp=[rois, gt_boxes, gt_cls],
            #                                                        Tout=[tf.float32, tf.float32, tf.float32])   
            roi_bbox_targets = tf.stop_gradient(roi_bbox_targets)
            roi_gt_labels = tf.stop_gradient(roi_gt_labels)
            
            return rois, roi_bbox_targets, roi_gt_labels, rpn_labels, rpn_bbox_targets
        
        else:
            
            rois = self.proposal_layer(rpn_cls_prob, self.rpn_bbox_pred, 
                                       anchors, img_size)
            
            return rois
        
    @tf.function(experimental_relax_shapes=True)
    def rpn_body(self, feature_map):
        """
        Region Proposal Network body utilizes feature map extracted from base net
        and generate object classification predictions and bounding box deltas on anchors.
        
        Inputs:
        - feature_map: tf.float, feature map from base net, shape=(n, h, w, c)
        
        Outputs:
        - anchors: tf.float, anchor bounding boxes based on feature map size and anchor reference. 
                   shape=(h*w*num_anchors, 4)
        - rpn_cls_score: tf.float, class raw scores for individual anchor bounding box.
                         shape=(n, h, w, num_anchors, 2)
        - rpn_cls_prob: tf.float, class probabilities after softmax, shape=(n, h, w, num_anchors, 2)
        - rpn_bbox_pred: tf.float, bounding box deltas, shape=(n, h, w, num_anchors, 4)
        - (h, w): tf.int, feature map height and width
        """
        
        # iterating over tf.Tensor is not allowed in Graph
        # n, h, w, _ = feature_map.shape
        
        shape = tf.shape(feature_map)
        n, h, w = shape[0], shape[1], shape[2]

        anchors = generate_anchors(self.anchors_ref, self.feat_stride, h, w)
        rpn = self.rpn_conv(feature_map)
        rpn_cls_score = self.score_conv(rpn)
        # rpn_cls_score_reshape: (n, h, w, num_anchors, 2)
        rpn_cls_score = tf.reshape(rpn_cls_score, (n, h, w, self.num_anchors, 2))
        rpn_cls_prob = self.softmax(rpn_cls_score)

        rpn_bbox_pred = self.bbox_conv(rpn)
        # (n, h, w, num_anchors, 4), (dx, dy, dw, dh)
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred, (n, h, w, self.num_anchors, -1))
        self.rpn_bbox_pred = rpn_bbox_pred
        
        return anchors, rpn_cls_prob, rpn_cls_score, rpn_bbox_pred, h, w
        