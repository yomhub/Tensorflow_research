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
    vgg16=tf.keras.applications.VGG16(
      input_tensor=tf.keras.Input(shape=input_shape[-3:]),
      weights='imagenet', 
      include_top=False)
    self.feature_model = tf.keras.Model(
      inputs=vgg16.inputs,
      outputs=[
        # vgg16.get_layer('block1_pool').output,
        # ch64 RF6*6 PIk2, after 3x3conv rf = 10
        # vgg16.get_layer('block2_conv2').output,
        # ch128 RF14*14 PIk2
        vgg16.get_layer('block2_pool').output,
        # ch128 RF16*16 PIk4, after 3x3conv rf = 24, fls = 8, flstr = 2
        # vgg16.get_layer('block3_conv3').output,
        # ch256 RF40*40 PIk4
        vgg16.get_layer('block3_pool').output,
        # ch256 RF44*44 PIk8
        # vgg16.get_layer('block4_conv3').output,
        # ch512 RF92*92 PIk8
        vgg16.get_layer('block4_pool').output,
        # ch512 RF100*100 PIk16
        # vgg16.get_layer('block5_conv3').output
        # ch512 RF196*196 PIk16
        vgg16.get_layer('block5_pool').output
        # ch512 RF212*212 PIk32
      ],
      name='vgg16'
    )

    # output[-1] ==> output[-2]
    # self.g1upc = tf.keras.layers.UpSampling2D(
    #   name = 'g1_UpSampling'
    # )
    self.g1upc = tf.keras.layers.Conv2DTranspose(
      filters = self.feature_model.output[-2].shape[-1], 
      kernel_size = (2, 2),
      strides = (2, 2),
      name = 'g1_upconv',
      )
    self.g1_c1 = tf.keras.layers.Conv2D(
      filters = self.feature_model.output[-2].shape[-1],
      kernel_size=(1, 1),
      activation=tf.nn.relu,
      name="g1_1x1_conv",
      padding="same",
    )
    self.g1_c2 = tf.keras.layers.Conv2D(
      filters = self.feature_model.output[-2].shape[-1],
      kernel_size=(3, 3),
      activation=tf.nn.relu,
      name="g1_1x1_conv",
      padding="same",
    )

    # output[-2] ==> output[-3]
    # self.g2upc = tf.keras.layers.UpSampling2D(
    #   name = 'g2_UpSampling'
    # )
    self.g2upc = tf.keras.layers.Conv2DTranspose(
      filters = self.feature_model.output[-3].shape[-1], 
      kernel_size = (2, 2),
      strides = (2, 2),
      name = 'g2_upconv',
      )
    self.g2_c1 = tf.keras.layers.Conv2D(
      filters = self.feature_model.output[-3].shape[-1],
      kernel_size=(1,1),
      activation=tf.nn.relu,
      name="g2_1x1_conv",
      padding="same",
    )
    self.g2_c2 = tf.keras.layers.Conv2D(
      filters = self.feature_model.output[-3].shape[-1],
      kernel_size=(3, 3),
      activation=tf.nn.relu,
      name="g2_1x1_conv",
      padding="same",
    )

    # output[-3] ==> output[-4]
    # self.g3upc = tf.keras.layers.UpSampling2D(
    #   name = 'g3_UpSampling'
    # )
    self.g3upc = tf.keras.layers.Conv2DTranspose(
      filters = self.feature_model.output[-4].shape[-1], 
      kernel_size = (2, 2),
      strides = (2, 2),
      name = 'g3_upconv',
      )
    self.g3_c1 = tf.keras.layers.Conv2D(
      filters = self.feature_model.output[-4].shape[-1],
      kernel_size=(1,1),
      activation=tf.nn.relu,
      name="g3_1x1_conv",
      padding="same",
    )
    self.g3_c2 = tf.keras.layers.Conv2D(
      filters = self.feature_model.output[-4].shape[-1],
      kernel_size=(3, 3),
      activation=tf.nn.relu,
      name="g3_1x1_conv",
      padding="same",
    )

    self.fin_conv = tf.keras.layers.Conv2D(
      filters = 32,
      kernel_size=(1,1),
      activation=tf.nn.relu,
      name="final_conv",
      padding="same",
    )
    self.map_conv = tf.keras.layers.Conv2D(
      filters = 2,
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
      [int(inputs.shape[-3]/32)*32,int(inputs.shape[-2]/32)*32])
    ftlist = self.feature_model(inputs)

    # output[-1] ==> output[-2]
    g1ft = self.g1upc(ftlist[-1])
    g1ft = tf.concat([g1ft,ftlist[-2]],axis=-1)
    g1ft = self.g1_c1(g1ft)
    g1ft = self.g1_c2(g1ft)

    # output[-2] ==> output[-3]
    g2ft = self.g2upc(g1ft)
    g2ft = tf.concat([g2ft,ftlist[-3]],axis=-1)
    g2ft = self.g2_c1(g2ft)
    g2ft = self.g2_c2(g2ft)

    # output[-3] ==> output[-4]
    g3ft = self.g3upc(g2ft)
    g3ft = tf.concat([g3ft,ftlist[-4]],axis=-1)
    g3ft = self.g3_c1(g3ft)
    g3ft = self.g3_c2(g3ft)

    fm = self.fin_conv(g3ft)
    fm = self.map_conv(fm)

    return fm

class UnetLoss(tf.keras.losses.Loss):
  """
    Args: los_mod in :
      'smp': sampel negtive points to number of postive points
      'nor': apply negtive/postive normalization on cross entropy
      'none' or None: directly use sparse_softmax_cross_entropy_with_logits
  """
  def __init__(self,los_mod = 'smp'):
    super(UnetLoss, self).__init__()
    self.los_mod = los_mod.lower() if(type(los_mod)==str and los_mod.lower() in ['smp','nor'])else 'none'
    
  def call(self, y_true, y_pred):
    y_true = tf.image.resize(y_true,y_pred.shape[-3:-1],'nearest')
    y_true = tf.reshape(tf.cast(tf.cast(y_true,tf.bool),tf.int64),[-1])
    y_pred = tf.reshape(y_pred,[-1,y_pred.shape[-1]])
    post = tf.where(y_true>0)[:,0]
    neg = tf.where(y_true<1)[:,0]

    if(self.los_mod=='nor'):
      loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.gather(y_pred,post), 
        labels=tf.gather(y_true,post)
        )) * (neg.shape[0]/y_true.shape[0])+ \
        tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.gather(y_pred,neg), 
        labels=tf.gather(y_true,neg)
        )) * (post.shape[0]/y_true.shape[0])
    elif(self.los_mod=='smp'):
      if(neg.shape[0]>post.shape[0]):
        neg = tf.gather(neg,
          tf.random.uniform([post.shape[0]],maxval=neg.shape[0]-1,dtype=tf.int32))
      loss = 0.7*tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.gather(y_pred,post), 
        labels=tf.gather(y_true,post)
        )) + 0.3*tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.gather(y_pred,neg), 
        labels=tf.gather(y_true,neg)
        ))
    else:
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
    return loss
