import tensorflow as tf
from model.vgg16 import VGG16
from model.rpn import RPN
from model.layer.roi_pooling_layer import roi_pooling
from model.tail import Tail
from config.config import cfg
from model.utils.bbox_transform import delta2bbox, clip_to_boundary
from tensorflow.keras.losses import sparse_categorical_crossentropy


class FasterRCNN(tf.keras.Model):
    """
    Faster R-CNN model 
                                                    / - labels
    input - base_net - rpn - tail - 
                                                    \ - bboxes
    For detailed architecture, please refer to README.md.
    The FasterRCNN class is constructed with 2 mode: 
    * Training mode for training model with specific datasets, evaluating and saving model.
    * Testing mode for inference with input as image or squence of images,
      output classes and bounding boxes.
    
    Args:
    - is_training: bool, Flag to switch network mode between training and testing.
    - num_classes: int, The number of classes in the dataset.
    - base_net: tf.keras.Model, The base net(or backbone) for extracting features.
    - rpn: tf.keras.Model, Region Proposal Network takes feature map and generates
                           proposal region of interests.
    - roi_pooling: tf.keras.Layer, Pooling RoIs in the feature map into the same size.
    - tail: tf.keras.Model, A model takes pooled RoIs, adjusts the bounding boxes
                            and classifies the labels of objects in RoIs.
    
    """
    
    def __init__(self, is_training):
        super(FasterRCNN, self).__init__()
        
        self.is_training = is_training
        self.num_classes = cfg.num_classes
        # initialize the layers here
        self.base_net = VGG16()
        self.rpn = RPN(self.is_training)
        self.roi_pooling = roi_pooling()
        self.tail = Tail(self.num_classes, self.is_training)
        
        # use ImageNet pre-trained weights to init vgg16 base net
        # this will be over written if checkpoint exsits
        if cfg.use_vgg_pretrain:
            self.base_net.build(input_shape=(None, None, None, 3))
            self.base_net.load_weights(self.base_net.model_path, by_name=True, skip_mismatch=True)
            self.tail.build(input_shape=(None, 7, 7, 512))
            self.tail.load_weights(self.base_net.model_path, by_name=True, skip_mismatch=True)
    
    def call(self, inputs):
        """
        Call method for Faster RCNN class.
        Inputs:
        - inputs: dict, A dict contains image info including image, gt_classes, gt_boxes and scale.
        
        Outputs:
        - total_loss: func, in training mode, returns the class loss and 
                            bouning box loss of RPN and ROI(tail).
        - labels, bboxes, cls_prob: tf.int, tf.float, tf.float,
                                    in test mode, the prediction of bouding boxes and classes. 
        
        """
        
        # unpack the inputs
        if self.is_training: # learning mode
            gt_boxes, gt_cls= inputs['gt_boxes'], inputs['gt_classes']
        img, img_size, scale = (inputs['img_scaled'],
                                inputs['scaled_size'],
                                inputs['scale'])
        
        feature_map = self.base_net(img)
        
        if self.is_training:
            (rois, roi_bbox_targets, roi_gt_labels, 
             rpn_labels, rpn_bbox_targets) = self.rpn(feature_map, img_size,
                                                      gt_boxes=gt_boxes, gt_cls=gt_cls)
        else:
            rois = self.rpn(feature_map, img_size)
            
        crops = self.roi_pooling(feature_map, rois, img_size)
        self.cls_scores, cls_prob, self.bbox_pred = self.tail(crops)
        labels = tf.argmax(cls_prob, axis=-1)
        
        if self.is_training:
            return self.total_loss(rpn_labels, rpn_bbox_targets, roi_bbox_targets, roi_gt_labels)
        
        # denormalize the bbox in testing
        if cfg.norm_bbox:
            box_pred = self.bbox_pred * cfg.norm_std + cfg.norm_mean
        
        # post progress for test/evaluation
        rois /= scale
        rois = tf.reshape(rois, (-1, 1, 4))
        rois = tf.broadcast_to(rois, box_pred.shape)

        bboxes = delta2bbox(tf.reshape(rois, (-1, 4)), 
                            tf.reshape(box_pred, (-1, 4)))

        bboxes = clip_to_boundary(bboxes, inputs['img_size'])
        bboxes = tf.reshape(bboxes, (-1, self.num_classes, 4))
        
        return labels, bboxes, cls_prob
    
    def total_loss(self, rpn_labels, rpn_bbox_targets, roi_bbox_targets, roi_gt_labels):
        """
        Calculate rpn loss and roi loss
        
        Notation:
        - (n, h, w, _): feature map shape
        - S: number of sampled rois(training)
        - num_anchors: number of anchors per anchor point
        
        Input:
        - rpn_labels: tf.int, labels of anchors, shape=(n, h, w, num_anchors, 1)
        - rpn_bbox_targets: tf.float, shape=(n, h, w, num_anchors, 4)
        - roi_bbox_targets: tf.float, bbox regression targets, shape=(S, 4)
        - roi_gt_labels: tf.int, sampled roi labels, shape=(S, 1)
    
        Output:
        - rpn_cls_loss: tf.float, region proposal network class loss
        - rpn_bbox_loss: tf.float, region proposal network bbox loss
        - roi_cls_loss: tf.float, rcnn class loss
        - roi_bbox_loss: tf.float, rcnn bbox loss
        
        """
        # ---------------------------- rpn losses -------------------------------- #
        # select the labels which are not -1
        rpn_select = tf.where(tf.math.not_equal(rpn_labels, -1))
        rpn_labels_select = tf.reshape(tf.gather_nd(rpn_labels, rpn_select), (-1, 1))
        rpn_cls_score = tf.gather_nd(self.rpn.rpn_cls_score, rpn_select[:, :-1])
        rpn_cls_loss = sparse_categorical_crossentropy(y_true=rpn_labels_select, # (256, 1)
                                                       y_pred=rpn_cls_score, # (256, 2)
                                                       from_logits=True)
        rpn_cls_loss = tf.reduce_mean(rpn_cls_loss)
        
        rpn_bbox_loss = _bbox_loss(self.rpn.rpn_bbox_pred, rpn_bbox_targets, rpn_labels, 3.0)
        
        # ----------------------------- roi losses -------------------------------- #
        roi_cls_loss = sparse_categorical_crossentropy(y_true=roi_gt_labels,# (128, 1)
                                                       y_pred=self.cls_scores, # (128, 21)
                                                       from_logits=True)
        roi_cls_loss = tf.reduce_mean(roi_cls_loss)
        roi_select = tf.stack([tf.range(0, cfg.num_sample_rois), 
                               tf.reshape(tf.cast(roi_gt_labels, tf.int32), (-1, ))])
        roi_select = tf.transpose(roi_select) # (128, 2)
        box_pred = tf.gather_nd(self.bbox_pred, roi_select)
        roi_bbox_loss = _bbox_loss(box_pred, roi_bbox_targets, roi_gt_labels, 1.0)
        
        return rpn_cls_loss, rpn_bbox_loss, roi_cls_loss, roi_bbox_loss


def _smooth_l1_loss(pred_box, gt_box, mask, sigma):
    
    sigma2 = sigma ** 2
    # ATTENTION: ALWAYS KEEP TRACK OF THE TENSOR SHAPE
    if mask.shape[1] > 1:
        diff = tf.gather_nd(pred_box - gt_box, mask[:, :-1])
    else:
        diff = tf.gather_nd(pred_box - gt_box, mask[:])
    diff_abs = tf.abs(diff)
    sign = tf.stop_gradient(tf.cast(tf.less(diff_abs, 1. / sigma2), dtype=tf.float32))
    loss = tf.pow(diff, 2) * (sigma2 / 2.) * sign + (diff_abs - (0.5 / sigma2)) * (1. - sign)
    
    return loss


def _bbox_loss(pred_box, gt_box, labels, sigma):
    
    # only regress the foreground
    mask = tf.where(labels > 0)
    loss = _smooth_l1_loss(pred_box, gt_box, mask, sigma)
    
    # normalize the regression loss by the number of pos and neg samples
    # norm should be the number of rpn samples(256) and roi samples(128),
    # since fix the total sampling number in to_target layers.
    norm = tf.cast(tf.where(labels >= 0).shape[0], tf.float32)
    
    return tf.reduce_sum(loss) / norm
