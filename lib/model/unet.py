from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
import math
from types import *
from tflib.bbox_transform import map2coordinate, yxyx2xywh, map2coordinate

class Unet(tf.keras.Model):
  """
    Args:
      std: True to apply per_image_standardization
    Outputs: dictionary of
      {
        'mask': edge mask with [1,h,w,1]
        'gt':gtbox with [1,h,w,4] where 4 is [center_x,center_y,w,h]
          in [0,1]
        'scr': score map with [1,h,w,2]
      }
  """
  def __init__(self,std=True):
    super(Unet, self).__init__()
    self.std = bool(std)
  
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
      filters = 1, # predict the value of map because sobel filter is float
      kernel_size=(3, 3),
      activation=tf.nn.relu,
      # activation=None,
      name="map_conv",
      padding="same",
    )
    self.scr_conv = tf.keras.layers.Conv2D(
      filters = 2, 
      kernel_size=(3, 3),
      activation=tf.nn.relu,
      # activation=None,
      name="scr_conv",
      padding="same",
    )
    self.box_conv = tf.keras.layers.Conv2D(
      filters = 4, # [cx,cy,w,h] in [0,1]
      kernel_size=(3, 3),
      activation=tf.nn.relu,
      # activation=None,
      name="box_conv",
      padding="same",
    )
    super(Unet, self).build(input_shape)

  def call(self, inputs):
    inputs = tf.image.resize(
      inputs,
      [int(inputs.shape[-3]/self.max_scale_factor)*self.max_scale_factor,int(inputs.shape[-2]/self.max_scale_factor)*self.max_scale_factor])
    if(self.std):inputs = tf.image.per_image_standardization(inputs)
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

    ft = self.fin_conv(ft)
    mask = self.map_conv(ft)
    gts = self.box_conv(ft)
    scr = self.scr_conv(ft)

    return {'mask':mask,'gt':gts,'scr':scr}

class UnetLoss(tf.keras.losses.Loss):
  """
    Assume y_true is binary pixel map.
    y_true: {
      'gt': (n,4 or 5) shape tensor where 4 or 5 is 
      [(label,) y1, x1, y2, x2] coordinate in [0,1]
      'mask': image tensor with value 0-255
    }
    Args: 
      los_mod in :
        'smp': sampel negtive points to number of postive points
        'nor': apply negtive/postive normalization on cross entropy
        'none' or None: directly use sparse_softmax_cross_entropy_with_logits
      shr: shrunk rate for box in [0-0.5]
  """
  def __init__(self,los_mod = 'smp',bd_enf = False, shr=0.2):
    super(UnetLoss, self).__init__()
    self.los_mod = los_mod.lower() if(type(los_mod)==str and los_mod.lower() in ['smp','nor'])else 'none'
    self.shr_co = float(shr) if(shr<0.5 and shr>=0.0)else 0.2
  
  def mask_loss(self, y_true, y_pred, gtbox):
    y_true = tf.image.resize(y_true,y_pred.shape[-3:-1],'nearest')
    gtbox = map2coordinate(gtbox,[1.0,1.0],y_pred.shape[-3:-1])
    gtbox = tf.cast(yxyx2xywh(gtbox,center=False),tf.int32)
    loss = 0.0
    for i in range(gtbox.shape[0]):
      s_yt = tf.image.crop_to_bounding_box(y_true,gtbox[i][-3],gtbox[i][-4],gtbox[i][-1],gtbox[i][-2])
      s_yp = tf.image.crop_to_bounding_box(y_pred,gtbox[i][-3],gtbox[i][-4],gtbox[i][-1],gtbox[i][-2])
      s_yt = tf.reshape(tf.cast(s_yt,tf.float64),[-1])
      s_yp = tf.reshape(tf.cast(s_yp,tf.float64),[-1])
      loss += 1.0 - 2.0*tf.math.reduce_sum(s_yt*s_yp)/tf.math.reduce_sum(s_yt*s_yt+s_yp*s_yp)
    loss /= gtbox.shape[0]

    return tf.cast(loss,tf.float32)

  def box_score_loss(self, gtbox, y_pred_gt, y_pred_scr):
    loss = 0.0
    gtbox = yxyx2xywh(gtbox)
    sh_gtbox = tf.stack([
      gtbox[:,-4]*y_pred_scr.shape[-2], # cx
      gtbox[:,-3]*y_pred_scr.shape[-3], # cy
      # shrunk box by self.shr_co 
      gtbox[:,-2]*y_pred_scr.shape[-2]*self.shr_co, # w
      gtbox[:,-1]*y_pred_scr.shape[-3]*self.shr_co], # h
      axis=1)
    csxs = tf.cast(tf.math.floor(sh_gtbox[:,-4]),tf.int32)
    cexs = tf.cast(tf.math.ceil(sh_gtbox[:,-4]+1),tf.int32)
    csys = tf.cast(tf.math.floor(sh_gtbox[:,-3]),tf.int32)
    ceys = tf.cast(tf.math.ceil(sh_gtbox[:,-3]+1),tf.int32)

    # box loss
    label = np.zeros(y_pred_scr.shape[-3:-1],dtype=np.int32)
    for i in range(gtbox.shape[0]):
      label[csys[i]:ceys[i],csxs[i]:cexs[i]] = 1
      loss += tf.reduce_mean(
        tf.reduce_sum(tf.math.abs(y_pred_gt[0,csys[i]:ceys[i],csxs[i]:cexs[i],:]-gtbox[i,-4:]),axis=-1)
        )
    loss = tf.math.divide(loss,gtbox.shape[0])

    label = tf.cast(tf.reshape(label,[-1]),tf.int64)
    y_pred_scr = tf.reshape(y_pred_scr,[-1,y_pred_scr.shape[-1]])
    post = tf.where(label>0)[:,0]
    neg = tf.where(label<1)[:,0]

    if(self.los_mod=='nor'):
      loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.gather(y_pred_scr,post), 
        labels=tf.gather(label,post)
        )) * (neg.shape[0]/label.shape[0])+ \
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.gather(y_pred_scr,neg), 
        labels=tf.gather(label,neg)
        )) * (post.shape[0]/label.shape[0])
    elif(self.los_mod=='smp'):
      if(neg.shape[0]>post.shape[0]):
        neg = tf.gather(neg,
          tf.random.uniform([post.shape[0]],maxval=neg.shape[0]-1,dtype=tf.int32))
      loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.gather(y_pred_scr,post), 
        labels=tf.gather(label,post)
        )) + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.gather(y_pred_scr,neg), 
        labels=tf.gather(label,neg)
        ))
    else:
      loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred_scr, labels=label))

    return tf.cast(loss,tf.float32)

  def call(self, y_true, y_pred):
    loss = 0.0
    loss += self.mask_loss(y_true['mask'],y_pred['mask'],y_true['gt'][:,-4:])
    loss += self.box_score_loss(gtbox=y_true['gt'][:,-4:],y_pred_gt=y_pred['gt'],y_pred_scr=y_pred['scr'])

    return loss
