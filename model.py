#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import division
import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim
import blocks.FSM as FSM
from blocks.FSM import unpool as unpool

tf.app.flags.DEFINE_string('network', 'resnet_v1_50', '')
tf.app.flags.DEFINE_integer('text_scale', 512, '')
tf.app.flags.DEFINE_integer('hard_pos_samples', 128, '')
tf.app.flags.DEFINE_integer('rand_pos_samples', 128, '')
tf.app.flags.DEFINE_integer('hard_neg_samples', 512, '')
tf.app.flags.DEFINE_integer('rand_neg_samples', 512, '')
tf.app.flags.DEFINE_boolean('use_FSM', True, 'use_FSM')
from nets import nets_factory

FLAGS = tf.app.flags.FLAGS

def unpool_hw(inputs,h=2,w=2):
    if h == 1 and w == 1:
        return inputs
    #if h<1 or w<1:
        #size_ = [int(1/h)]
        #return tf.nn.max_pool(inputs, ksize=[], strides=[], padding="SAME")
    if h <1 and w<1:
        size_ = [int(1/h), int(1/w)]
        return tf.nn.max_pool(inputs, ksize=size_, strides=size_, padding="SAME")
    elif h<1:
        size_ = [int(1/h), 1]
        mid_ = tf.nn.max_pool(inputs, ksize=size_, strides=size_, padding="SAME")
        if w >1:
            return tf.image.resize_bilinear(mid_, size=[tf.shape(mid_)[1],  tf.shape(mid_)[2]*w])
        else:
            return mid_
    elif w<1:
        size_ = [1, int(1/w)]
        mid_ = tf.nn.max_pool(inputs, ksize=size_, strides=size_, padding="SAME")
        if h >1:
            return tf.image.resize_bilinear(mid_, size=[tf.shape(mid_)[1]*h,  tf.shape(mid_)[2]])
        else:
            return mid_
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*h,  tf.shape(inputs)[2]*w])


def np_max_pool(feature_map, stride_h=2, stride_w=2):
    batch = feature_map.shape[0]
    height=feature_map.shape[1]
    width=feature_map.shape[2]
    channel=feature_map.shape[3]
    if stride_h == 1:
        feature_map = feature_map.reshape(batch, height, int(width/stride_w), stride_w, channel)
        feature_map = np.max(feature_map,axis=3)
    elif stride_w == 1:
        feature_map = feature_map.reshape(batch, int(height/stride_h), stride_h, width, channel)
        feature_map = np.max(feature_map,axis=2)
    else:
        feature_map = feature_map.reshape(batch, int(height/stride_h), stride_h, int(width/stride_w), stride_w, channel)
        feature_map = np.max(feature_map,axis=2)
        feature_map = np.max(feature_map,axis=3)
    return feature_map
    
    #if h <1 and w<1:
        #h_ = np.int32(1/h); w_=np.int32(1/w)#h_ = tf.cast(1/h,dtype=tf.int32); w_ = tf.cast(1/w,dtype=tf.int32)
        #print("2:",w_,h_)
        #return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]/h_,  tf.shape(inputs)[2]/w_])
    #elif h<1:
        #h_ = np.int32(1/h)#tf.cast(1/h,dtype=tf.int32);
        #print("3:",w,h_)
        #return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]/h_,  tf.shape(inputs)[2]*w])
    #elif w<1:
        #w_ = np.int32(1/w)#tf.cast(1/w,dtype=tf.int32)
        #print("4:",w_,h)
        #return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*h,  tf.shape(inputs)[2]/w_])
        
def unpool_hw_stride(inputs,h=2,w=2):
    if h == 1 and w == 1:
        return inputs
    if h <1 and w<1:
        return inputs[:, ::int(1/h), ::int(1/w),:]
    elif h<1:
        mid_ = inputs[:, ::int(1/h), :,:]
        if w >1:
            return tf.image.resize_bilinear(mid_, size=[tf.shape(mid_)[1],  tf.shape(mid_)[2]*w])
        else:
            return mid_
    elif w<1:
        mid_ = inputs[:, :, ::int(1/w),:]
        if h >1:
            return tf.image.resize_bilinear(mid_, size=[tf.shape(mid_)[1]*h,  tf.shape(mid_)[2]])
        else:
            return mid_
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*h,  tf.shape(inputs)[2]*w])

def np_max_pool_stride(feature_map, stride_h=2, stride_w=2):
    return feature_map[:, ::stride_h , ::stride_w, :]

def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

#anchor_sizes = [32, 64, 128, 256, 512, 1024]
#f_fscale_index = [0, 0, 1, 2, 3, 3]
#f_scale_i = [1, 1, 2, 4, 8, 8]
def model(images, anchor_sizes, f_fscale_index, f_scale_i, select_split_N, is_select_background=False, weight_decay=1e-5, is_training=True, use_2branch=True ):
    '''
    define the model, we use slim's implemention of resnet
    '''
    images = mean_image_subtraction(images)

    base_network = nets_factory.get_network_fn(FLAGS.network, num_classes=2, weight_decay=weight_decay, is_training=is_training)
    logits, end_points = base_network(images)
    #with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
    #    logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            if  FLAGS.use_FSM:
                p5_fsm32 = FSM.FSM_func_32(end_points['pool5'],'pool5')
                p4_fsm16, p4_fsm32 = FSM.FSM_func_16(end_points['pool4'],'pool4')
                p3_fsm8,  p3_fsm16, p3_fsm32 = FSM.FSM_func_8(end_points['pool3'],'pool3')
                p2_fsm4,  p2_fsm8,  p2_fsm16, p2_fsm32 = FSM.FSM_func_4(end_points['pool2'],'pool2')
            
                f = [tf.concat([p5_fsm32, p4_fsm32, p3_fsm32, p2_fsm32], axis=-1), tf.concat([p4_fsm16, p3_fsm16, p2_fsm16], axis=-1), tf.concat([p3_fsm8, p2_fsm8], axis=-1), p2_fsm4]
            else:
                f = [end_points['pool5'], end_points['pool4'],
                    end_points['pool3'], end_points['pool2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None]
            h = [None, None, None, None]
            num_outputs = [96, 64, 48, 32]
            for i in range(4):
                if i == 0:
                    h[i] = f[i]
                else:
                    c1_1 = slim.conv2d(tf.concat([unpool(h[i-1]), f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
            # for i in range(4):
            #     if i == 0:
            #         h[i] = f[i]
            #     else:
            #         c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
            #         h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
            #     if i <= 2:
            #         g[i] = unpool(h[i])
            #     else:
            #         g[i] = slim.conv2d(h[i], num_outputs[i], 3)
            #     print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

            r_g3 = g[3]
            r_g2 = g[2]
            r_g1 = g[1]
            r_g0 = g[0]
                

            
            g3h = [None, None, None, None]; g3w = [None, None, None, None]
            g2h = [None, None, None, None]; g2w = [None, None, None, None]
            g1h = [None, None, None, None]; g1w = [None, None, None, None]
            g0h = [None, None, None, None]; g0w = [None, None, None, None]

            #################r_g3 4*4
            g3h[0] = slim.conv2d(r_g3, 8, 1)#4*4
            g3h[1] = unpool_hw_stride(slim.conv2d(r_g3, 8, 1),h=1/2,w=1)#8*4
            g3h[2] = unpool_hw_stride(slim.conv2d(r_g3, 8, 1),h=1/4,w=1)#16*4
            g3h[3] = unpool_hw_stride(slim.conv2d(r_g3, 8, 1),h=1/8,w=1)#32*4
            g3w[0] = slim.conv2d(r_g3, 8, 1)#4*4
            g3w[1] = unpool_hw_stride(slim.conv2d(r_g3, 8, 1),h=1,w=1/2)#4*8
            g3w[2] = unpool_hw_stride(slim.conv2d(r_g3, 8, 1),h=1,w=1/4)#4*16
            g3w[3] = unpool_hw_stride(slim.conv2d(r_g3, 8, 1),h=1,w=1/8)#4*32
            #################r_g2 8*8
            g2h[0] = unpool_hw_stride(slim.conv2d(r_g2, 8, 1),h=2,w=2)#4*4
            g2h[1] = unpool_hw_stride(slim.conv2d(r_g2, 8, 1),h=1,w=2)#8*4
            g2h[2] = unpool_hw_stride(slim.conv2d(r_g2, 8, 1),h=1/2,w=2)#16*4
            g2h[3] = unpool_hw_stride(slim.conv2d(r_g2, 8, 1),h=1/4,w=2)#32*4
            g2w[0] = unpool_hw_stride(slim.conv2d(r_g2, 8, 1),h=2,w=2)#4*4
            g2w[1] = unpool_hw_stride(slim.conv2d(r_g2, 8, 1),h=2,w=1)#4*8
            g2w[2] = unpool_hw_stride(slim.conv2d(r_g2, 8, 1),h=2,w=1/2)#4*16
            g2w[3] = unpool_hw_stride(slim.conv2d(r_g2, 8, 1),h=2,w=1/4)#4*32
            #################r_g1 16*16
            g1h[0] = unpool_hw_stride(slim.conv2d(r_g1, 8, 1),h=4,w=4)#4*4
            g1h[1] = unpool_hw_stride(slim.conv2d(r_g1, 8, 1),h=2,w=4)#8*4
            g1h[2] = unpool_hw_stride(slim.conv2d(r_g1, 8, 1),h=1,w=4)#16*4
            g1h[3] = unpool_hw_stride(slim.conv2d(r_g1, 8, 1),h=1/2,w=4)#32*4
            g1w[0] = unpool_hw_stride(slim.conv2d(r_g1, 8, 1),h=4,w=4)#4*4
            g1w[1] = unpool_hw_stride(slim.conv2d(r_g1, 8, 1),h=4,w=2)#4*8
            g1w[2] = unpool_hw_stride(slim.conv2d(r_g1, 8, 1),h=4,w=1)#4*16
            g1w[3] = unpool_hw_stride(slim.conv2d(r_g1, 8, 1),h=4,w=1/2)#4*32
            #################r_g0 16*16
            g0h[0] = unpool_hw_stride(slim.conv2d(r_g0, 8, 1),h=8,w=8)#4*4
            g0h[1] = unpool_hw_stride(slim.conv2d(r_g0, 8, 1),h=4,w=8)#8*4
            g0h[2] = unpool_hw_stride(slim.conv2d(r_g0, 8, 1),h=2,w=8)#16*4
            g0h[3] = unpool_hw_stride(slim.conv2d(r_g0, 8, 1),h=1,w=8)#32*4
            g0w[0] = unpool_hw_stride(slim.conv2d(r_g0, 8, 1),h=8,w=8)#4*4
            g0w[1] = unpool_hw_stride(slim.conv2d(r_g0, 8, 1),h=8,w=4)#4*8
            g0w[2] = unpool_hw_stride(slim.conv2d(r_g0, 8, 1),h=8,w=2)#4*16
            g0w[3] = unpool_hw_stride(slim.conv2d(r_g0, 8, 1),h=8,w=1)#4*32
            
            f_h = [None, None, None, None, None, None]#32 64 /128 /256 /512 1024
            f_w = [None, None, None, None, None, None]
            for fi in range(len(f_h)):
                i = f_fscale_index[fi]
                rate_ = [1,1]
                if i == 1 or i == 5:
                    rate_ = [2,2]
                f_h[fi] = slim.conv2d(slim.conv2d(tf.concat([g3h[i],g2h[i],g1h[i],g0h[i]],axis=-1), 16, 3, rate=rate_), 16, 3, rate=rate_)
                f_w[fi] = slim.conv2d(slim.conv2d(tf.concat([g3w[i],g2w[i],g1w[i],g0w[i]],axis=-1), 16, 3, rate=rate_), 16, 3, rate=rate_)
                
                
            f_h_cls = [None, None, None, None, None, None]#32 64 /128 /256 /512 1024
            f_h_reg = [None, None, None, None, None, None]
            f_h_ratio = [None, None, None, None, None, None]
            f_h_rt = [None, None, None, None, None, None] #regress top
            f_h_rb = [None, None, None, None, None, None] #regress botoom
            f_h_angle = [None, None, None, None, None, None]
            
            f_w_cls = [None, None, None, None, None, None]#32 64 /128 /256 /512 1024
            f_w_reg = [None, None, None, None, None, None]
            f_w_ratio = [None, None, None, None, None, None]
            f_w_rl = [None, None, None, None, None, None] #regress left
            f_w_rr = [None, None, None, None, None, None] #regress right
            f_w_angle = [None, None, None, None, None, None]
            
            for fi in range(len(anchor_sizes)):
                f_h_cls[fi] = slim.conv2d(slim.conv2d(f_h[fi], 4, 3), 1, 1, activation_fn=None, normalizer_fn=None)
                f_h_reg[fi] = tf.exp(slim.conv2d(slim.conv2d(f_h[fi], 4, 3), 1, 1, activation_fn=None, normalizer_fn=None)) * anchor_sizes[fi]
                f_h_ratio[fi] = slim.conv2d(slim.conv2d(f_h[fi], 4, 3), 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                f_h_rt[fi] = f_h_reg[fi] * f_h_ratio[fi]
                f_h_rb[fi] = f_h_reg[fi] * (1 - f_h_ratio[fi])
                f_h_angle[fi] = (slim.conv2d(slim.conv2d(f_h[fi], 4, 3), 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2
                
                f_w_cls[fi] = slim.conv2d(slim.conv2d(f_w[fi], 4, 3), 1, 1, activation_fn=None, normalizer_fn=None)
                f_w_reg[fi] = tf.exp(slim.conv2d(slim.conv2d(f_w[fi], 4, 3), 1, 1, activation_fn=None, normalizer_fn=None)) * anchor_sizes[fi]
                f_w_ratio[fi] = slim.conv2d(slim.conv2d(f_w[fi], 4, 3), 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                f_w_rl[fi] = f_w_reg[fi] * f_w_ratio[fi]
                f_w_rr[fi] = f_w_reg[fi] * (1 - f_w_ratio[fi])
                f_w_angle[fi] = (slim.conv2d(slim.conv2d(f_w[fi], 4, 3), 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2
            

            f_cls_list_h = []; f_cls_list_w = []
            f_rl_list = []; f_rr_list = []; f_rt_list = []; f_rb_list = []
            f_angle_list_h  = []; f_angle_list_w  = []
            if use_2branch:
                f_all_list  = []
            for fi in range(len(anchor_sizes)):
                if use_2branch:
                    f_all_list.append(unpool_hw_stride(f_h[fi],h=f_scale_i[fi], w=1))
                    f_all_list.append(unpool_hw_stride(f_w[fi],h=1, w=f_scale_i[fi]))
                f_cls_list_h.append(unpool_hw_stride(f_h_cls[fi],h=f_scale_i[fi], w=1))
                f_cls_list_w.append(unpool_hw_stride(f_w_cls[fi],h=1, w=f_scale_i[fi]))
                
            f_cls_list_h_loss_pos = tf.concat(f_cls_list_h,axis=-1)
            f_cls_list_w_loss_pos = tf.concat(f_cls_list_w,axis=-1)
            f_cls_list_h_loss = tf.concat([f_cls_list_h_loss_pos, slim.conv2d(slim.conv2d(f_cls_list_h_loss_pos,1,3),1,1, activation_fn=None, normalizer_fn=None)],axis=-1)
            f_cls_list_w_loss = tf.concat([f_cls_list_w_loss_pos, slim.conv2d(slim.conv2d(f_cls_list_w_loss_pos,1,3),1,1, activation_fn=None, normalizer_fn=None)],axis=-1)
            fm_N = tf.concat([f_cls_list_w_loss, f_cls_list_h_loss], axis=-1)
            
            f_cls_list_h_weight = tf.split(tf.nn.softmax(f_cls_list_h_loss, axis=-1), len(anchor_sizes)+1, axis=-1)
            f_cls_list_w_weight = tf.split(tf.nn.softmax(f_cls_list_w_loss, axis=-1), len(anchor_sizes)+1, axis=-1)
            for fi in range(len(anchor_sizes)):
                f_rt_list.append(unpool_hw_stride(f_h_rt[fi],h=f_scale_i[fi], w=1) * (f_cls_list_h_weight[fi]))
                f_rb_list.append(unpool_hw_stride(f_h_rb[fi],h=f_scale_i[fi], w=1) * (f_cls_list_h_weight[fi]))
                f_rl_list.append(unpool_hw_stride(f_w_rl[fi],h=1, w=f_scale_i[fi]) * (f_cls_list_w_weight[fi]))
                f_rr_list.append(unpool_hw_stride(f_w_rr[fi],h=1, w=f_scale_i[fi]) * (f_cls_list_w_weight[fi]))
                
                f_angle_list_h.append(unpool_hw_stride(f_h_angle[fi],h=f_scale_i[fi], w=1) * (f_cls_list_h_weight[fi]))
                f_angle_list_w.append(unpool_hw_stride(f_w_angle[fi],h=1, w=f_scale_i[fi]) * (f_cls_list_w_weight[fi]))
            
            

            
            F_score = (tf.reduce_sum(tf.concat(f_cls_list_h_weight[0:len(anchor_sizes)], axis=-1), keep_dims=True, axis=-1) \
                + tf.reduce_sum(tf.concat(f_cls_list_w_weight[0:len(anchor_sizes)], axis=-1), keep_dims=True, axis=-1))/2
            
            f_rl = tf.reduce_sum(tf.concat(f_rl_list, axis=-1), keep_dims=True, axis=-1)
            f_rr = tf.reduce_sum(tf.concat(f_rr_list, axis=-1), keep_dims=True, axis=-1)
            f_rt = tf.reduce_sum(tf.concat(f_rt_list, axis=-1), keep_dims=True, axis=-1)
            f_rb = tf.reduce_sum(tf.concat(f_rb_list, axis=-1), keep_dims=True, axis=-1)
            f_angle_h = tf.reduce_sum(tf.concat(f_angle_list_h, axis=-1), keep_dims=True, axis=-1)
            f_angle_w = tf.reduce_sum(tf.concat(f_angle_list_w, axis=-1), keep_dims=True, axis=-1)
            f_angle = (f_angle_h + f_angle_w) /2
            
            F_geometry = tf.concat([f_rt, f_rr, f_rb, f_rl,f_angle], axis=-1)
            
            if use_2branch:
                f_all = slim.conv2d(slim.conv2d(tf.concat(f_all_list, axis=-1), 32 ,3), 32 ,3)
                F_score_full = slim.conv2d(slim.conv2d(f_all, 2, 3), 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                f_box_full_only = tf.exp(slim.conv2d(slim.conv2d(f_all, 8, 3), 4, 1, activation_fn=None, normalizer_fn=None)) * FLAGS.text_scale
                f_angle_full = (slim.conv2d(slim.conv2d(f_all, 2, 3), 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2
                F_geometry_full = tf.concat([f_box_full_only, f_angle_full], axis=-1)
            #F_geometry_full_only = tf.concat([f_box_full_only, f_angle_full], axis=-1)
            
            #F_geometry_full = (F_geometry * F_score + F_geometry_full_only * F_score_full) / (F_score + F_score_full)
            
            #F_cls_map = tf.concat(f_cls_list_h, axis=-1) + tf.concat(f_cls_list_w, axis=-1)
            if use_2branch:
                return F_score, F_geometry, fm_N, F_score_full, F_geometry_full
            return F_score, F_geometry, fm_N, F_score, F_geometry




