from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
import math
from types import *
from model.config import cfg
from tflib.bbox_transform import *
from tflib.anchor_target_layer import anchor_target_layer_tf, point_anchor_target_layer_tf
from tflib.proposal_target_layer import proposal_target_layer_tf
from tflib.snippets import generate_real_anchors, score_convert
from tflib.common import *

class Unet(tf.keras.Model):
  def __init__(self):
    super(Unet, self).__init__()
  
  def build(self,input_shape):
    imgh = input_shape[-3]
    imgw = input_shape[-2]
    res50 = tf.keras.applications.ResNet50(
      # input_tensor=tf.keras.Input(shape=input_shape[-3:]),
      weights='imagenet', 
      include_top=False)
    self.max_scale_factor = 16
    self.feature_model = tf.keras.Model(
      inputs=res50.inputs,
      outputs=[
        res50.get_layer('pool1_pool').output,
        # 1/4, ch=64
        res50.get_layer('conv2_block3_out').output,
        # 1/4, ch=256
        res50.get_layer('conv3_block4_out').output,
        # 1/8, ch=512
        res50.get_layer('conv4_block6_out').output,
        # 1/16, ch=1024
        # res50.get_layer('conv5_block3_out').output
        # 1/32, ch=2048
      ],
      name='res50'
    )
    g1_g3_chs_list = [128,64,32]
    # output[-1] ==> output[-2]
    self.g0upc = tf.keras.layers.UpSampling2D(
      name = 'g0_UpSampling'
    )
    # output[-2] ==> output[-3]
    self.g1_c1 = tf.keras.layers.Conv2D(
      filters = g1_g3_chs_list[0],
      kernel_size=(1, 1),
      activation=tf.nn.relu,
      name="g1_1x1_conv",
      padding="same",
    )
    self.g1_c2 = tf.keras.layers.Conv2D(
      filters = g1_g3_chs_list[0],
      kernel_size=(3, 3),
      activation=tf.nn.relu,
      name="g1c2_3x3_conv",
      padding="same",
    )
    self.g1upc = tf.keras.layers.UpSampling2D(
      name = 'g1_UpSampling'
    )
    # self.g1upc = tf.keras.layers.Conv2DTranspose(
    #   filters = g1_g3_chs_list[0], 
    #   kernel_size = (3, 3),
    #   strides = (3, 3),
    #   use_bias = False,
    #   activation = 'relu',
    #   name = 'g1_upconv',
    #   )


    # output[-3] ==> output[-4]
    self.g2_c1 = tf.keras.layers.Conv2D(
      filters = g1_g3_chs_list[1],
      kernel_size=(1,1),
      activation=tf.nn.relu,
      name="g2c1_1x1_conv",
      padding="same",
    )
    self.g2_c2 = tf.keras.layers.Conv2D(
      filters = g1_g3_chs_list[1],
      kernel_size=(3, 3),
      activation=tf.nn.relu,
      name="g2c2_3x3_conv",
      padding="same",
    )
    # self.g2upc = tf.keras.layers.UpSampling2D(
    #   name = 'g2_UpSampling'
    # )
    # self.g2upc = tf.keras.layers.Conv2DTranspose(
    #   filters = g1_g3_chs_list[1], 
    #   kernel_size = (2, 2),
    #   strides = (2, 2),
    #   use_bias = False,
    #   activation = 'relu',
    #   name = 'g2_upconv',
    #   )


    # output[-3] ==> output[-4]
    self.g3_c1 = tf.keras.layers.Conv2D(
      filters = g1_g3_chs_list[2],
      kernel_size=(1,1),
      activation=tf.nn.relu,
      name="g3c1_1x1_conv",
      padding="same",
    )
    self.g3_c2 = tf.keras.layers.Conv2D(
      filters = g1_g3_chs_list[2],
      kernel_size=(3, 3),
      activation=tf.nn.relu,
      name="g3c2_3x3_conv",
      padding="same",
    )
    self.g3_c3 = tf.keras.layers.Conv2D(
      filters = g1_g3_chs_list[2],
      kernel_size=(3, 3),
      activation=tf.nn.relu,
      name="g3c3_1x1_conv",
      padding="same",
    )
    # self.g3upc = tf.keras.layers.UpSampling2D(
    #   name = 'g3_UpSampling'
    # )
    # self.g3upc = tf.keras.layers.Conv2DTranspose(
    #   filters = g1_g3_chs_list[2], 
    #   kernel_size = (2, 2),
    #   strides = (2, 2),
    #   use_bias = False,
    #   activation = 'relu',
    #   name = 'g3_upconv',
    #   )

    # process list between each output
    self.fet_proc_list=[
      [self.g0upc],
      [self.g1_c1,self.g1_c2,self.g1upc],
      [self.g2_c1,self.g2_c2],
      [self.g3_c1,self.g3_c2,self.g3_c3],
    ]

    self.fin_conv = tf.keras.layers.Conv2D(
      filters = 32,
      kernel_size=(1,1),
      activation=tf.nn.relu,
      name="final_conv",
      padding="same",
    )
    self.map_conv = tf.keras.layers.Conv2D(
      filters = 1, # [bg, pixel in boundary, pixel inside stroke]
      kernel_size=(3, 3),
      activation=tf.nn.relu,
      # activation=None,
      name="map_conv",
      padding="same",
    )
    super(Unet, self).build(input_shape)

  def call(self, inputs):
    inputs = tf.image.resize(
      inputs,
      [int(inputs.shape[-3]/self.max_scale_factor)*self.max_scale_factor,int(inputs.shape[-2]/self.max_scale_factor)*self.max_scale_factor])
    ftlist = self.feature_model(inputs)

    # output[-1] ==> output[-2]
    ft = self.g0upc(ftlist[-1])
    ft = tf.concat([ft,ftlist[-2]],axis=-1)

    # output[-2] ==> output[-3]
    ft = self.g1_c1(ft)
    ft = self.g1_c2(ft)
    ft = self.g1upc(ft)
    ft = tf.concat([ft,ftlist[-3]],axis=-1)

    # output[-3] ==> output[-4]
    ft = self.g2_c1(ft)
    ft = self.g2_c2(ft)
    # ft = self.g2upc(ft)
    ft = tf.concat([ft,ftlist[-4]],axis=-1)

    # output[-4] ==> output[-5]
    ft = self.g3_c1(ft)
    ft = self.g3_c2(ft)
    ft = self.g3_c3(ft)

    fm = self.fin_conv(ft)
    fm = self.map_conv(fm)

    return fm

class UnetLoss(tf.keras.losses.Loss):
  """
    Assume y_true is binary pixel map.
    Args: 
      los_mod in :
        'smp': sampel negtive points to number of postive points
        'nor': apply negtive/postive normalization on cross entropy
        'none' or None: directly use sparse_softmax_cross_entropy_with_logits
  """
  def __init__(self,los_mod = 'smp',bd_enf = False):
    super(UnetLoss, self).__init__()
    self.los_mod = los_mod.lower() if(type(los_mod)==str and los_mod.lower() in ['smp','nor'])else 'none'
    
  def call(self, y_true, y_pred):
    y_true = tf.image.resize(y_true,y_pred.shape[-3:-1],'nearest')
    # y_true = tf.cast(tf.cast(y_true,tf.bool),tf.float64) # y_true in [0,1]
    y_true = tf.reshape(y_true,[-1])
    y_pred = tf.reshape(y_pred,[-1])

    # post = tf.where(y_true>0)[:,0]
    # neg = tf.where(y_true<1)[:,0]

    # if(self.los_mod=='nor'):
    #   loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=tf.gather(y_pred,post), 
    #     labels=tf.gather(y_true,post)
    #     )) * (neg.shape[0]/y_true.shape[0])+ \
    #     tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=tf.gather(y_pred,neg), 
    #     labels=tf.gather(y_true,neg)
    #     )) * (post.shape[0]/y_true.shape[0])
    # elif(self.los_mod=='smp'):
    #   if(neg.shape[0]>post.shape[0]):
    #     neg = tf.gather(neg,
    #       tf.random.uniform([post.shape[0]],maxval=neg.shape[0]-1,dtype=tf.int32))
    #   loss = 0.7*tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=tf.gather(y_pred,post), 
    #     labels=tf.gather(y_true,post)
    #     )) + 0.3*tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=tf.gather(y_pred,neg), 
    #     labels=tf.gather(y_true,neg)
    #     ))
    # else:
    #   loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
    loss = 1.0 - 2*tf.math.reduce_sum(y_true*y_pred_b)/tf.math.reduce_sum(y_true*y_true+y_pred*y_pred)

    return loss
