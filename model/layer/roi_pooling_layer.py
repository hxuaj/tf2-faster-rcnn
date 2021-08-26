import tensorflow as tf
from tensorflow.keras import layers
from config.config import cfg


class roi_pooling(layers.Layer):
    """
    ROI Pooling Layer
    Extracts crops the feature map from base net with region of interests
    and resizes them to fixes size.
    
    Args:
    - pool_size: int, the crop height and width after extracting from feature map
    
    """
    
    def __init__(self):
        super(roi_pooling, self).__init__()
        
        self.pool_size = cfg.pool_size
        self.max_pool = layers.MaxPool2D((2, 2), padding='same')
        
    @tf.function(experimental_relax_shapes=True)
    def call(self, feature_map, rois, img_size):
        """
        Inputs:
        - feature_map: tf.float, the feature map extracted by base net, shape=(n, h, w, c)
        - rois: tf.float, region of interests, shape=(N, 4)
        - img_size: tf.int, input scaled image size
        
        Outputs:
        - crops: tf.float, shape=(N, pool_size, pool_size, c)
        
        """

        # Normalize the bbox coordinates
        # image coordinates maximum, range: [0, image_size - 1]
        img_h = tf.cast(img_size[0], dtype=tf.float32) - 1.0
        img_w = tf.cast(img_size[1], dtype=tf.float32) - 1.0
        
        x1 = tf.slice(rois, [0, 0], [-1, 1]) / img_w
        y1 = tf.slice(rois, [0, 1], [-1, 1]) / img_h
        x2 = tf.slice(rois, [0, 2], [-1, 1]) / img_w
        y2 = tf.slice(rois, [0, 3], [-1, 1]) / img_h
        # concat order according to tf.image.crop_and_resize
        boxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
        # boxes = tf.concat([y1, x1, y2, x2], axis=1)
        # since all the rois come from the same image
        N = tf.shape(rois)[0]
        box_indices = tf.zeros((N, ), dtype=tf.int32)
        box_indices = tf.stop_gradient(box_indices)
        
        if cfg.if_max_pool:
            crop_size = self.pool_size * 2
            crops = tf.image.crop_and_resize(feature_map, boxes, box_indices, 
                                             [crop_size, crop_size])
            
            return self.max_pool(crops)
        else:
        
            return tf.image.crop_and_resize(feature_map, boxes, box_indices, 
                                             [self.pool_size, self.pool_size])