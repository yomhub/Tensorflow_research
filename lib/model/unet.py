from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
import math
from types import *
from tflib.bbox_transform import map2coordinate, yxyx2xywh, map2coordinate
from config.unet_default import cfg

class Unet(tf.keras.Model):
  """
    Args:
      std: True to apply per_image_standardization
    Outputs: dictionary of
      {
        'mask': edge mask with [1,h,w,1] in value [0,1]
        'gt':gtbox with [1,h,w,4] where 4 is [center_x,center_y,w,h]
          in [0,1]
        'scr': score map with [1,h,w,2]
      }
  """
  def __init__(self,std=True,feat=cfg['F_NET']):
    super(Unet, self).__init__()
    self.std = bool(std)
    self.feat = feat.lower() if(feat.lower() in ['vgg16','resnet'])else 'vgg16'
    self.src_chs = cfg['SRC_CHS']
  
  def build(self,input_shape):
    if(self.feat=='resnet'):
      fnet = tf.keras.applications.ResNet50(
        # input_tensor=tf.keras.Input(shape=input_shape[-3:]),
        weights=cfg['F_NET_PT'], 
        include_top=False)      
    else:
      fnet = tf.keras.applications.VGG16(
        # input_tensor=tf.keras.Input(shape=input_shape[-3:]),
        weights=cfg['F_NET_PT'], 
        include_top=False)

    self.feature_model = tf.keras.Model(
      inputs=fnet.inputs,
      outputs=[fnet.get_layer(o).output for o in cfg['F_LAYER'][self.feat]],
      name=self.feat
    )
    self.max_scale_factor = cfg['F_MXS'][self.feat]
    f_layer_num = len(self.feature_model.outputs)

    self.scor_map_list = [
      tf.keras.layers.Conv2D(
        filters = self.src_chs,
        kernel_size=(1, 1),
        activation = cfg['SRC_ACT'],
        name="l{}_score_map".format(i),
        padding="same",
      )
      for i in range(f_layer_num)]

    self.rect_list = [
      tf.keras.layers.Conv2D(
        filters = self.feature_model.outputs[i].shape[-1],
        kernel_size=(1, 1),
        activation=tf.nn.relu,
        name="{}to{}_1x1_conv".format(i+1,i),
        padding="same",
      )
      for i in range(f_layer_num-2,-1,-1)]

    self.rect_list.append(tf.keras.layers.Conv2D(
      filters = cfg['F_FCHS'][self.feat],
      kernel_size=(1, 1),
      activation=tf.nn.relu,
      name="fin_1x1_conv",
      padding="same",
    ))

    self.map_conv = tf.keras.layers.Conv2D(
      filters = 1, # predict the value of map because sobel filter is float
      kernel_size=(3, 3),
      activation=tf.nn.relu,
      # activation=None,
      name="map_conv",
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

    ft = None
    scr = None
    for i in range(len(ftlist)):
      if(scr!=None):scr = tf.math.add(tf.image.resize(scr,ftlist[-1-i].shape[-3:-1]),self.scor_map_list[i](ftlist[-1-i]))
      else:scr = self.scor_map_list[i](ftlist[-1-i])
      if(ft!=None):ft = self.rect_list[i](tf.math.add(tf.image.resize(ft,ftlist[-1-i].shape[-3:-1]),ftlist[-1-i]))
      else:ft = self.rect_list[i](ftlist[-1-i])

    mask = self.map_conv(ft)
    gts = self.box_conv(ft)

    self.ft = ft
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
      part_mask_loss: True to enable partial mask loss
  """
  def __init__(self,los_mod = 'nor',bd_enf = False, shr=0.2, part_mask_loss=True):
    super(UnetLoss, self).__init__()
    self.los_mod = los_mod.lower() if(type(los_mod)==str and los_mod.lower() in ['smp','nor'])else 'none'
    self.shr_co = float(shr) if(shr<0.5 and shr>=0.0)else 0.2
    self.cur_loss={}
    self.part_mask_loss = bool(part_mask_loss)
  
  def mask_loss(self, y_ture_mask, y_pred, gtbox):
    y_ture_mask = tf.image.resize(y_ture_mask,y_pred.shape[-3:-1],'nearest')
    y_ture_mask = tf.norm(tf.image.sobel_edges(y_ture_mask),axis=-1)
    y_ture_mask /= 255.0
    gtbox = map2coordinate(gtbox,[1.0,1.0],y_pred.shape[-3:-1])
    gtbox = tf.cast(yxyx2xywh(gtbox,center=False),tf.int32)
    loss = 0.0
    if(self.part_mask_loss):
      c=0.0
      for i in range(gtbox.shape[0]):
        if(gtbox[i][-1]<=0 or gtbox[i][-2]<=0):continue
        try:
          s_yt = tf.image.crop_to_bounding_box(y_ture_mask,gtbox[i][-3],gtbox[i][-4],gtbox[i][-1],gtbox[i][-2])
          s_yp = tf.image.crop_to_bounding_box(y_pred,gtbox[i][-3],gtbox[i][-4],gtbox[i][-1],gtbox[i][-2])
          s_yt = tf.reshape(tf.cast(s_yt,tf.float64),[-1])
          s_yp = tf.reshape(tf.cast(s_yp,tf.float64),[-1])
          loss += 1.0 - 2.0*tf.math.reduce_sum(s_yt*s_yp)/tf.math.reduce_sum(s_yt*s_yt+s_yp*s_yp)
          c+=1.0
        except:
          continue
      loss /= c
    else:
      y_ture_mask = tf.cast(y_ture_mask,y_pred.dtype)
      loss += 1.0 - 2.0*tf.math.reduce_sum(y_ture_mask*y_pred)/tf.math.reduce_sum(y_ture_mask*y_ture_mask+y_pred*y_pred)

    return tf.cast(loss,tf.float32)

  def pixel_clf_loss(self, y_ture_mask, y_pred_scr):
    y_ture_mask = tf.cast(tf.cast(y_ture_mask,tf.bool),tf.int64)
    y_ture_mask = tf.reshape(y_ture_mask,[-1])
    y_pred_scr = tf.reshape(y_pred_scr,[-1,y_pred_scr.shape[-1]])

    pos_pix = tf.where(y_ture_mask==1)
    pos_num = int(pos_pix.shape[0])
    neg_num = min(int(pos_num*cfg['NP_RATE']),y_pred_scr[y_ture_mask==0].shape[0])
    neg_num = min(y_pred_scr[y_ture_mask==0].shape[0],cfg['PX_CLS_LOSS_MAX_NEG'])
    vals, _ = tf.math.top_k(y_pred_scr[y_ture_mask==0][:,0],k=neg_num) # negative
    neg_slc = tf.cast(tf.math.logical_and(y_pred_scr[:,0]>=vals[-1],y_ture_mask==0),tf.float32)
    weight = neg_slc+float(cfg['PX_CLS_LOSS_W'])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred_scr, labels=y_ture_mask)
    loss = tf.reduce_sum(weight*loss)/float(pos_num+neg_num)

    return tf.cast(loss,tf.float32)

  def box_score_loss(self, gtbox, y_pred_gt, y_pred_scr):
    bxloss = 0.0
    scloss = 0.0
    gtbox = yxyx2xywh(gtbox,True)
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
      bxloss += tf.reduce_mean(
        tf.reduce_sum(tf.math.abs(y_pred_gt[0,csys[i]:ceys[i],csxs[i]:cexs[i],:]-gtbox[i,-4:]),axis=-1)
        )
    bxloss = tf.math.divide(bxloss,gtbox.shape[0])

    # label = tf.cast(tf.reshape(label,[-1]),tf.int64)
    # y_pred_scr = tf.reshape(y_pred_scr,[-1,y_pred_scr.shape[-1]])
    # post = tf.where(label>0)[:,0]
    # neg = tf.where(label<1)[:,0]

    # if(self.los_mod=='nor'):
    #   scloss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=tf.gather(y_pred_scr,post), 
    #     labels=tf.gather(label,post)
    #     )) * (neg.shape[0]/label.shape[0])+ \
    #     tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=tf.gather(y_pred_scr,neg), 
    #     labels=tf.gather(label,neg)
    #     )) * (post.shape[0]/label.shape[0])
    # elif(self.los_mod=='smp'):
    #   if(neg.shape[0]>post.shape[0]):
    #     neg = tf.gather(neg,
    #       tf.random.uniform([post.shape[0]],maxval=neg.shape[0]-1,dtype=tf.int32))
    #   scloss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=tf.gather(y_pred_scr,post), 
    #     labels=tf.gather(label,post)
    #     )) + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=tf.gather(y_pred_scr,neg), 
    #     labels=tf.gather(label,neg)
    #     ))
    # else:
    #   scloss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred_scr, labels=label))

    return tf.cast(bxloss,tf.float32) #,tf.cast(scloss,tf.float32)

  def call(self, y_true, y_pred):
    mask = tf.image.resize(y_pred['mask'],y_pred['scr'].shape[-3:-1],'nearest')
    mskloss = self.mask_loss(y_true['mask'],mask,y_true['gt'][:,-4:])
    scloss = self.pixel_clf_loss(y_ture_mask=mask, y_pred_scr=y_pred['scr'])
    bxloss = self.box_score_loss(gtbox=y_true['gt'][:,-4:],y_pred_gt=y_pred['gt'],y_pred_scr=y_pred['scr'])
    self.cur_loss={
      'mask':mskloss,
      'box':bxloss,
      'score':scloss,
      }
    return 0.1*mskloss + 0.1*bxloss + scloss
