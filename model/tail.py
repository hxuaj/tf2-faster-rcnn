import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten, Dense, Dropout, Softmax
from config.config import cfg


class Tail(tf.keras.Model):
    """
    The Tail of the Model including two fully connected layers, a classification FC layer and
    a bounding box regression FC layer.
    
    """
    
    def __init__(self, num_classes, is_training):
        super(Tail, self).__init__()
        
        self.is_training = is_training
        # Args
        self.units = 4096
        self.num_classes = num_classes
        self.weight_decay = cfg.weight_decay
        self.regularizer = tf.keras.regularizers.l2(self.weight_decay)
        
        # Layers
        self.flatten = Flatten(name='flatten')
        self.dense1 = Dense(self.units, activation='relu', kernel_regularizer=self.regularizer, name='fc1')
        self.dense2 = Dense(self.units, activation='relu', kernel_regularizer=self.regularizer, name='fc2')
        self.dropout1 = Dropout(rate=0.5)
        self.dropout2 = Dropout(rate=0.5)
        self.dense_to_class = Dense(self.num_classes, kernel_regularizer=self.regularizer)
        self.dense_to_box = Dense(self.num_classes * 4, kernel_regularizer=self.regularizer)
        self.softmax = Softmax(axis=-1)
    
    @tf.function(experimental_relax_shapes=True)
    def call(self, crops):
        """
        Call method for Tail.
        
        Notation:
        - N: number of proposed rois(different at training and testing)
        - pool_size: the crop height and width after extracting from feature map
        - c: number of feature map channels
        
        Inputs:
        - crops: tf.float, portions of feature map extracted with rois. shape=(N, pool_size, pool_size, c)
        
        Outputs:
        - cls_scores: tf.float, class raw scores of crops, shape=(N, num_classes)
        - cls_prob: tf.float, class probabilies, shape=(N, num_classes)
        - bbox_pred: tf.float, bounding boxes delta predictions, shape(N, num_classes, 4)
        
        """
        
        crop_flat = self.flatten(crops) # shape=(crops.shape[0], -1)
        fc1 = self.dense1(crop_flat)
        
        if self.is_training:
            fc1 = self.dropout1(fc1, training=True)

        fc2 = self.dense2(fc1) # shape=(crops.shape[0], self.units)
        
        if self.is_training:
            fc2 = self.dropout2(fc2, training=True)
        
        cls_scores = self.dense_to_class(fc2)
        cls_prob = self.softmax(cls_scores)
        # cls_pred = tf.argmax(cls_prob, axis=-1)
        bbox_pred = self.dense_to_box(fc2)
        bbox_pred = tf.reshape(bbox_pred, (-1, self.num_classes, 4))
        
        return cls_scores, cls_prob, bbox_pred

