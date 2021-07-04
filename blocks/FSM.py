import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])

def FSM_func_4(feature, name='', middle_channels = 32, scales=[1,1,1,1]):
    feature_32 = slim.conv2d(feature, middle_channels, 1, padding='SAME',activation_fn=None)#

    branch0_0 = slim.conv2d(feature_32, middle_channels, 3, padding='SAME')#, scope=name+"_0", reuse=tf.AUTO_REUSE)
    branch0_01 = slim.conv2d(branch0_0, middle_channels, 3, padding='SAME')#, scope=name+"_01", reuse=tf.AUTO_REUSE)
    branch0_1 = slim.conv2d(branch0_01, middle_channels, 3, padding='SAME',activation_fn=None)#, scope=name+"_1", reuse=tf.AUTO_REUSE)*scales[0]
                    
    branch1_0_down = slim.max_pool2d(feature_32, [2,2], stride=2, padding='SAME')
    branch1_0 = slim.conv2d(branch1_0_down, middle_channels, [3,3], rate=[1,1], padding='SAME')#, scope=name+"_0", reuse=tf.AUTO_REUSE)
    branch1_01 = slim.conv2d(branch1_0, middle_channels, [3,3], rate=[1,1], padding='SAME')#, scope=name+"_01", reuse=tf.AUTO_REUSE)
    branch1_1 = slim.conv2d(branch1_01, middle_channels, [3,3], rate=[1,1], padding='SAME',activation_fn=None)#, scope=name+"_1", reuse=tf.AUTO_REUSE)*scales[1]
    branch1_up = unpool(branch1_1)
    
                    
    branch2_0_down = slim.max_pool2d(feature_32, [4,4], stride=4, padding='SAME')
    branch2_0 = slim.conv2d(branch2_0_down, middle_channels, [3,3], rate=[1,1], padding='SAME')#, scope=name+"_0", reuse=tf.AUTO_REUSE)
    branch2_01 = slim.conv2d(branch2_0, middle_channels, [3,3], rate=[1,1], padding='SAME')#, scope=name+"_01", reuse=tf.AUTO_REUSE)
    branch2_1 = slim.conv2d(branch2_01, middle_channels, [3,3], rate=[1,1], padding='SAME',activation_fn=None)#, scope=name+"_1", reuse=tf.AUTO_REUSE)*scales[2]
    branch2_up_0 = unpool(branch2_1)
    branch2_up = unpool(branch2_up_0)

    branch3_0_down = slim.max_pool2d(feature_32, [8,8], stride=8, padding='SAME')
    branch3_0 = slim.conv2d(branch3_0_down, middle_channels, [3,3], rate=[1,1], padding='SAME')#, scope=name+"_0", reuse=tf.AUTO_REUSE)
    branch3_01 = slim.conv2d(branch3_0, middle_channels, [3,3], rate=[1,1], padding='SAME')#, scope=name+"_01", reuse=tf.AUTO_REUSE)
    branch3_1 = slim.conv2d(branch3_01, middle_channels, [3,3], rate=[1,1], padding='SAME',activation_fn=None)#, scope=name+"_1", reuse=tf.AUTO_REUSE)*scales[3]
    branch3_up_0 = unpool(branch3_1)
    branch3_up_1 = unpool(branch3_up_0)
    branch3_up = unpool(branch3_up_1)

    #feature_32 = tf.expand_dims(feature_32,axis=-1)
    #branch0_1 = tf.expand_dims(branch0_1,axis=-1)
    #branch1_up = tf.expand_dims(branch1_up,axis=-1)
    #branch2_up = tf.expand_dims(branch2_up,axis=-1)
    #branch3_up = tf.expand_dims(branch3_up,axis=-1)
    #down_up_s = tf.concat([feature_32, branch0_1, branch1_up, branch2_up, branch3_up],axis=-1)
    #down_up_ = tf.reduce_max(down_up_s,axis=-1)
    down_up_ = tf.nn.relu(feature_32 + branch0_1 + branch1_up + branch2_up + branch3_up)
    branch_8_ = tf.nn.relu(branch1_1 + branch2_up_0 + branch3_up_1)
    branch_16_ = tf.nn.relu(branch2_1 + branch3_up_0)
    branch_32_ = tf.nn.relu(branch3_1)
    return down_up_, branch_8_, branch_16_, branch_32_

def FSM_func_8(feature, name='', middle_channels = 32, scales=[1,1,1,1]):
    feature_32 = slim.conv2d(feature, middle_channels, 1, padding='SAME',activation_fn=None)

    branch0_0 = slim.conv2d(feature_32, middle_channels, 3, padding='SAME')#, scope=name+"_0", reuse=tf.AUTO_REUSE)
    branch0_01 = slim.conv2d(branch0_0, middle_channels, 3, padding='SAME')#, scope=name+"_01", reuse=tf.AUTO_REUSE)
    branch0_1 = slim.conv2d(branch0_01, middle_channels, 3, padding='SAME',activation_fn=None)#, scope=name+"_1", reuse=tf.AUTO_REUSE)*scales[0]
                    
    branch1_0_down = slim.max_pool2d(feature_32, [2,2], stride=2, padding='SAME')
    branch1_0 = slim.conv2d(branch1_0_down, middle_channels, [3,3], rate=[1,1], padding='SAME')#, scope=name+"_0", reuse=tf.AUTO_REUSE)
    branch1_01 = slim.conv2d(branch1_0, middle_channels, [3,3], rate=[1,1], padding='SAME')#, scope=name+"_01", reuse=tf.AUTO_REUSE)
    branch1_1 = slim.conv2d(branch1_01, middle_channels, [3,3], rate=[1,1], padding='SAME',activation_fn=None)#, scope=name+"_1", reuse=tf.AUTO_REUSE)*scales[1]
    branch1_up = unpool(branch1_1)
    
                    
    branch2_0_down = slim.max_pool2d(feature_32, [4,4], stride=4, padding='SAME')
    branch2_0 = slim.conv2d(branch2_0_down, middle_channels, [3,3], rate=[1,1], padding='SAME')#, scope=name+"_0", reuse=tf.AUTO_REUSE)
    branch2_01 = slim.conv2d(branch2_0, middle_channels, [3,3], rate=[1,1], padding='SAME')#, scope=name+"_01", reuse=tf.AUTO_REUSE)
    branch2_1 = slim.conv2d(branch2_01, middle_channels, [3,3], rate=[1,1], padding='SAME',activation_fn=None)#, scope=name+"_1", reuse=tf.AUTO_REUSE)*scales[2]
    branch2_up_0 = unpool(branch2_1)
    branch2_up = unpool(branch2_up_0)

    branch3_0_down = slim.max_pool2d(feature_32, [4,4], stride=4, padding='SAME')
    branch3_0 = slim.conv2d(branch3_0_down, middle_channels, [3,3], rate=[2,2], padding='SAME')#, scope=name+"_0", reuse=tf.AUTO_REUSE)
    branch3_01 = slim.conv2d(branch3_0, middle_channels, [3,3], rate=[2,2], padding='SAME')#, scope=name+"_01", reuse=tf.AUTO_REUSE)
    branch3_1 = slim.conv2d(branch3_01, middle_channels, [3,3], rate=[2,2], padding='SAME',activation_fn=None)#, scope=name+"_1", reuse=tf.AUTO_REUSE)*scales[3]
    branch3_up_0 = unpool(branch3_1)
    branch3_up = unpool(branch3_up_0)

    #feature_32 = tf.expand_dims(feature_32,axis=-1)
    #branch0_1 = tf.expand_dims(branch0_1,axis=-1)
    #branch1_up = tf.expand_dims(branch1_up,axis=-1)
    #branch2_up = tf.expand_dims(branch2_up,axis=-1)
    #branch3_up = tf.expand_dims(branch3_up,axis=-1)
    #down_up_s = tf.concat([feature_32, branch0_1, branch1_up, branch2_up, branch3_up],axis=-1)
    #down_up_ = tf.reduce_max(down_up_s,axis=-1)
    down_up_ = tf.nn.relu(feature_32 + branch0_1 + branch1_up + branch2_up + branch3_up)
    branch_16_ = tf.nn.relu(branch1_1 + branch2_up_0 + branch3_up_0)
    branch_32_ = tf.nn.relu(branch2_1 + branch3_1)
    return down_up_, branch_16_, branch_32_

def FSM_func_16(feature, name='', middle_channels = 32, scales=[1,1,1,1]):
    feature_32 = slim.conv2d(feature, middle_channels, 1, padding='SAME',activation_fn=None)

    branch0_0 = slim.conv2d(feature_32, middle_channels, 3, padding='SAME')#, scope=name+"_0", reuse=tf.AUTO_REUSE)
    branch0_01 = slim.conv2d(branch0_0, middle_channels, 3, padding='SAME')#, scope=name+"_01", reuse=tf.AUTO_REUSE)
    branch0_1 = slim.conv2d(branch0_01, middle_channels, 3, padding='SAME',activation_fn=None)#, scope=name+"_1", reuse=tf.AUTO_REUSE)*scales[0]
                    
    branch1_0_down = slim.max_pool2d(feature_32, [2,2], stride=2, padding='SAME')
    branch1_0 = slim.conv2d(branch1_0_down, middle_channels, [3,3], rate=[1,1], padding='SAME')#, scope=name+"_0", reuse=tf.AUTO_REUSE)
    branch1_01 = slim.conv2d(branch1_0, middle_channels, [3,3], rate=[1,1], padding='SAME')#, scope=name+"_01", reuse=tf.AUTO_REUSE)
    branch1_1 = slim.conv2d(branch1_01, middle_channels, [3,3], rate=[1,1], padding='SAME',activation_fn=None)#, scope=name+"_1", reuse=tf.AUTO_REUSE)*scales[1]
    branch1_up = unpool(branch1_1)
    
                    
    branch2_0_down = slim.max_pool2d(feature_32, [2,2], stride=2, padding='SAME')
    branch2_0 = slim.conv2d(branch2_0_down, middle_channels, [3,3], rate=[2,2], padding='SAME')#, scope=name+"_0", reuse=tf.AUTO_REUSE)
    branch2_01 = slim.conv2d(branch2_0, middle_channels, [3,3], rate=[2,2], padding='SAME')#, scope=name+"_01", reuse=tf.AUTO_REUSE)
    branch2_1 = slim.conv2d(branch2_01, middle_channels, [3,3], rate=[2,2], padding='SAME',activation_fn=None)#, scope=name+"_1", reuse=tf.AUTO_REUSE)*scales[2]
    branch2_up = unpool(branch2_1)

    branch3_0_down = slim.max_pool2d(feature_32, [2,2], stride=2, padding='SAME')
    branch3_0 = slim.conv2d(branch3_0_down, middle_channels, [3,3], rate=[4,4], padding='SAME')#, scope=name+"_0", reuse=tf.AUTO_REUSE)
    branch3_01 = slim.conv2d(branch3_0, middle_channels, [3,3], rate=[4,4], padding='SAME')#, scope=name+"_01", reuse=tf.AUTO_REUSE)
    branch3_1 = slim.conv2d(branch3_01, middle_channels, [3,3], rate=[4,4], padding='SAME',activation_fn=None)#, scope=name+"_1", reuse=tf.AUTO_REUSE)*scales[3]
    branch3_up = unpool(branch3_1)

    #feature_32 = tf.expand_dims(feature_32,axis=-1)
    #branch0_1 = tf.expand_dims(branch0_1,axis=-1)
    #branch1_up = tf.expand_dims(branch1_up,axis=-1)
    #branch2_up = tf.expand_dims(branch2_up,axis=-1)
    #branch3_up = tf.expand_dims(branch3_up,axis=-1)
    #down_up_s = tf.concat([feature_32, branch0_1, branch1_up, branch2_up, branch3_up],axis=-1)
    #down_up_ = tf.reduce_max(down_up_s,axis=-1)
    down_up_ = tf.nn.relu(feature_32 + branch0_1 + branch1_up + branch2_up + branch3_up)
    branch_32_ = tf.nn.relu(branch1_1 + branch2_1 + branch3_1)
    return down_up_, branch_32_

def FSM_func_32(feature, name='', middle_channels = 32, scales=[1,1,1,1]):
    feature_32 = slim.conv2d(feature, middle_channels, 1, padding='SAME',activation_fn=None)

    branch0_0 = slim.conv2d(feature_32, middle_channels, 3, padding='SAME')#, scope=name+"_0", reuse=tf.AUTO_REUSE)
    branch0_01 = slim.conv2d(branch0_0, middle_channels, 3, padding='SAME')#, scope=name+"_01", reuse=tf.AUTO_REUSE)
    branch0_1 = slim.conv2d(branch0_01, middle_channels, 3, padding='SAME')#, scope=name+"_1", reuse=tf.AUTO_REUSE)*scales[0]
                    
    branch1_0 = slim.conv2d(feature_32, middle_channels, [3,3], rate=[2,2], padding='SAME')#, scope=name+"_0", reuse=tf.AUTO_REUSE)
    branch1_01 = slim.conv2d(branch1_0, middle_channels, [3,3], rate=[2,2], padding='SAME')#, scope=name+"_01", reuse=tf.AUTO_REUSE)
    branch1_1 = slim.conv2d(branch1_01, middle_channels, [3,3], rate=[2,2], padding='SAME',activation_fn=None)#, scope=name+"_1", reuse=tf.AUTO_REUSE)*scales[1]
    
                    
    branch2_0 = slim.conv2d(feature_32, middle_channels, [3,3], rate=[4,4], padding='SAME')#, scope=name+"_0", reuse=tf.AUTO_REUSE)
    branch2_01 = slim.conv2d(branch2_0, middle_channels, [3,3], rate=[4,4], padding='SAME')#, scope=name+"_01", reuse=tf.AUTO_REUSE)
    branch2_1 = slim.conv2d(branch2_01, middle_channels, [3,3], rate=[4,4], padding='SAME',activation_fn=None)#, scope=name+"_1", reuse=tf.AUTO_REUSE)*scales[2]

    branch3_0 = slim.conv2d(feature_32, middle_channels, [3,3], rate=[8,8], padding='SAME')#, scope=name+"_0", reuse=tf.AUTO_REUSE)
    branch3_01 = slim.conv2d(branch3_0, middle_channels, [3,3], rate=[8,8], padding='SAME')#, scope=name+"_01", reuse=tf.AUTO_REUSE)
    branch3_1 = slim.conv2d(branch3_01, middle_channels, [3,3], rate=[8,8], padding='SAME',activation_fn=None)#, scope=name+"_1", reuse=tf.AUTO_REUSE)*scales[3]
    
    #feature_32 = tf.expand_dims(feature_32,axis=-1)
    #branch0_1 = tf.expand_dims(branch0_1,axis=-1)
    #branch1_1 = tf.expand_dims(branch1_1,axis=-1)
    #branch2_1 = tf.expand_dims(branch2_1,axis=-1)
    #branch3_1 = tf.expand_dims(branch3_1,axis=-1)
    #down_up_s = tf.concat([feature_32, branch0_1, branch1_1, branch2_1, branch3_1],axis=-1)
    #down_up_ = tf.reduce_max(down_up_s,axis=-1)    
    down_up_ = tf.nn.relu(feature_32 + branch0_1 + branch1_1 + branch2_1 + branch3_1)
    return down_up_
