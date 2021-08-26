import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.applications import VGG16
from config.config import cfg


class VGG16(tf.keras.Model):
    """
    Faster R-CNN's VGG16 base net implementation.
    The setup flows original paper with the first two blocks freezed by ImageNet pretrain model and
    no batch normalization is used.
    The training will backprop the rest of the network.
    
    """
    
    def __init__(self):
        super(VGG16, self).__init__()
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.root_path, 'vgg16.h5')
        self.weight_decay = cfg.weight_decay
        self.regularizer = tf.keras.regularizers.l2(self.weight_decay)

        # block 1
        self.conv1_1 = Conv2D(64, (3, 3), padding='same', activation='relu', 
                              trainable=False, name='block1_conv1')
        self.conv1_2 = Conv2D(64, (3, 3), padding='same', activation='relu', 
                              trainable=False, name='block1_conv2')
        self.maxpool1 = MaxPool2D(pool_size=(2, 2))
        # block 2
        self.conv2_1 = Conv2D(128, (3, 3), padding='same', activation='relu', 
                              trainable=False, name='block2_conv1')
        self.conv2_2 = Conv2D(128, (3, 3), padding='same', activation='relu', 
                              trainable=False, name='block2_conv2')
        self.maxpool2 = MaxPool2D(pool_size=(2, 2))
        # block 3
        self.conv3_1 = Conv2D(256, (3, 3), padding='same', activation='relu', 
                              kernel_regularizer=self.regularizer, name='block3_conv1')
        self.conv3_2 = Conv2D(256, (3, 3), padding='same', activation='relu', 
                              kernel_regularizer=self.regularizer, name='block3_conv2')
        self.conv3_3 = Conv2D(256, (3, 3), padding='same', activation='relu', 
                              kernel_regularizer=self.regularizer, name='block3_conv3')
        self.maxpool3 = MaxPool2D(pool_size=(2, 2))
        # block 4
        self.conv4_1 = Conv2D(512, (3, 3), padding='same', activation='relu', 
                              kernel_regularizer=self.regularizer, name='block4_conv1')
        self.conv4_2 = Conv2D(512, (3, 3), padding='same', activation='relu', 
                              kernel_regularizer=self.regularizer, name='block4_conv2')
        self.conv4_3 = Conv2D(512, (3, 3), padding='same', activation='relu', 
                              kernel_regularizer=self.regularizer, name='block4_conv3')
        self.maxpool4 = MaxPool2D(pool_size=(2, 2))
        # block 5
        self.conv5_1 = Conv2D(512, (3, 3), padding='same', activation='relu', 
                              kernel_regularizer=self.regularizer, name='block5_conv1')
        self.conv5_2 = Conv2D(512, (3, 3), padding='same', activation='relu', 
                              kernel_regularizer=self.regularizer, name='block5_conv2')
        self.conv5_3 = Conv2D(512, (3, 3), padding='same', activation='relu', 
                              kernel_regularizer=self.regularizer, name='block5_conv3')
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32)])
    def call(self, x):
        
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.maxpool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.maxpool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.maxpool3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.maxpool4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        
        return x
