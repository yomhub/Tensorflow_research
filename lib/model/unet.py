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
    self.f_layers_name = cfg['F_LAYER'][self.feat]
    self.fc6 = None
    self.fc6_cls = None
    self.fc7 = None
    self.fc7_cls = None
    self.fc_chs = 512
  
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
      outputs=[fnet.get_layer(o).output for o in self.f_layers_name if(not(o in ['fc6','fc7']))],
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
      )
      for i in range(f_layer_num)]

    self.rect_list = [
      tf.keras.layers.Conv2D(
        filters = self.feature_model.outputs[i].shape[-1],
        kernel_size=(1, 1),
        activation=tf.nn.relu,
        name="{}_to_{}_1x1_conv".format(i+1,i),
      )
      for i in range(f_layer_num-2,-1,-1)]

    if('fc6' in self.f_layers_name):
      self.fc6 = tf.keras.layers.Conv2D(filters = self.fc_chs, kernel_size=(1, 1), name='fc6')
      self.scor_map_list.insert(0,
        tf.keras.layers.Conv2D(
          filters = self.src_chs,
          kernel_size=(1, 1),
          activation = cfg['SRC_ACT'],
          name="fc6_score_map",
        ))
      self.rect_list.insert(0,tf.keras.layers.Conv2D(
        filters = self.feature_model.outputs[-1].shape[-1],
        kernel_size=(1, 1),
        activation=tf.nn.relu,
        name="fc6_to_{}_1x1_conv".format(f_layer_num-1),
      ))
    if('fc7' in self.f_layers_name):
      if(not('fc6' in self.f_layers_name)):
        self.fc6 = tf.keras.layers.Conv2D(filters = self.fc_chs, kernel_size=(1, 1), name='fc6')
        self.rect_list.insert(0,tf.keras.layers.Conv2D(
          filters = self.feature_model.outputs[-1].shape[-1],
          kernel_size=(1, 1),
          activation=tf.nn.relu,
          name="fc6_to_{}_1x1_conv".format(f_layer_num-1),
        ))
        self.scor_map_list.insert(0,None)  

      self.fc7 = tf.keras.layers.Conv2D(filters = self.fc_chs, kernel_size=(1, 1), name='fc7')
      self.scor_map_list.insert(0,
        tf.keras.layers.Conv2D(
          filters = self.src_chs,
          kernel_size=(1, 1),
          activation = cfg['SRC_ACT'],
          name="fc7_score_map",
        ))
      self.rect_list.insert(0,
        tf.keras.layers.Conv2D(
          filters = self.fc_chs,
          kernel_size=(1, 1),
          activation=tf.nn.relu,
          name="fc7_to_fc6_1x1_conv",
        )
      )
    
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
    if(self.std):inputs = tf.math.abs(tf.image.per_image_standardization(inputs))
    ftlist = self.feature_model(inputs)
    if(self.fc6):ftlist += [self.fc6(ftlist[-1])]
    if(self.fc7):ftlist += [self.fc7(ftlist[-1])]

    ft = None
    scr = None
    for i in range(len(ftlist)):
      if(self.scor_map_list[i]!=None):
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
    self.ls_fun = cfg['FIN_LOSS_FUN']
  
  def mask_loss(self, y_ture_mask, y_pred, gtbox):
    # y_ture_mask = tf.norm(tf.image.sobel_edges(y_ture_mask),axis=-1)
    y_ture_mask = tf.math.l2_normalize(y_ture_mask,axis=(-1,-2,-3))
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
          s_yt = tf.reshape(tf.cast(s_yt,tf.float32),[-1])
          s_yp = tf.reshape(tf.cast(s_yp,tf.float32),[-1])

          pos_num = tf.where(s_yt>0).shape[0]
          neg_num = min(int(cfg['MSK_LOSS_NP_RATE']*pos_num),int(s_yt.shape[0]-pos_num))
          vals, _ = tf.math.top_k(s_yp[s_yt==0],k=neg_num)
          pos_slc = tf.cast(s_yt>0,tf.float32)
          if(vals.shape[0]>0):
            neg_slc = tf.cast(tf.math.logical_and(s_yp>=vals[-1],s_yt==0.0),tf.float32)
          else:
            neg_slc = np.zeros_like(pos_slc)
          loss += tf.reduce_mean(tf.abs(s_yt-s_yp)*(cfg['MSK_LOSS_GLO_W']+cfg['MSK_LOSS_NG_W']*neg_slc+cfg['MSK_LOSS_PS_W']*pos_slc))
          # loss += 1.0 - 2.0*tf.math.reduce_sum(s_yt*s_yp)/tf.math.reduce_sum(s_yt*s_yt+s_yp*s_yp)
          c+=1.0
        except:
          continue
      if(c>0.0):loss /= c
      else:
        s_yt = tf.reshape(tf.cast(y_ture_mask,tf.float32),[-1])
        s_yp = tf.reshape(tf.cast(y_pred,tf.float32),[-1])
        pos_num = tf.where(s_yt>0).shape[0]
        neg_num = min(int(cfg['MSK_LOSS_NP_RATE']*pos_num),int(s_yt.shape[0]-pos_num))
        vals, _ = tf.math.top_k(s_yp[s_yt==0],k=neg_num)
        pos_slc = tf.cast(s_yt>0,tf.float32)
        if(vals.shape[0]>0):
          neg_slc = tf.cast(tf.math.logical_and(s_yp>=vals[-1],s_yt==0.0),tf.float32)
        else:
          neg_slc = np.zeros_like(pos_slc)
        loss = tf.reduce_mean(tf.abs(s_yt-s_yp)*(cfg['MSK_LOSS_GLO_W']+cfg['MSK_LOSS_NG_W']*neg_slc+cfg['MSK_LOSS_PS_W']*pos_slc))
    else:
      y_ture_mask = tf.cast(y_ture_mask,y_pred.dtype)
      loss += 1.0 - 2.0*tf.math.reduce_sum(y_ture_mask*y_pred)/tf.math.reduce_sum(y_ture_mask*y_ture_mask+y_pred*y_pred)

    return tf.cast(loss,tf.float32)

  def src_loss(self, y_ture_mask, y_pred_scr):
    # loss = (negtive pixel's weight + global weight)*CE
    y_ture_mask = tf.cast(tf.cast(y_ture_mask,tf.bool),tf.int64)
    y_ture_mask = tf.reshape(y_ture_mask,[-1])
    y_pred_scr = tf.reshape(y_pred_scr,[-1,y_pred_scr.shape[-1]])

    pos_pix = tf.where(y_ture_mask>0)
    pos_num = int(pos_pix.shape[0])
    neg_num = min(
      int(pos_num*cfg['PX_CLS_LOSS_NP_RATE']),
      y_pred_scr[y_ture_mask==0].shape[0],
      cfg['PX_CLS_LOSS_MAX_NEG']
      )
    vals, _ = tf.math.top_k(y_pred_scr[y_ture_mask==0][:,0],k=neg_num) # negative
    # Assign pixels in BG_prediction && BG_true with 1.0 weight
    neg_slc = tf.cast(tf.math.logical_and(y_pred_scr[:,0]>=vals[-1],y_ture_mask==0),tf.float32)
    pos_slc = tf.cast(y_ture_mask>0,tf.float32)
    
    weight = float(cfg['PX_CLS_LOSS_NG_W'])*neg_slc+float(cfg['PX_CLS_LOSS_PS_W'])*pos_slc+float(cfg['PX_CLS_LOSS_GLO_W'])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred_scr, labels=y_ture_mask)
    loss = tf.reduce_sum(weight*loss)/float(pos_num+neg_num)

    return tf.cast(loss,tf.float32)

  def box_loss(self, gtbox, y_pred_gt):
    bxloss = 0.0
    scloss = 0.0
    gtbox = yxyx2xywh(gtbox,True)
    sh_gtbox = tf.stack([
      gtbox[:,-4]*y_pred_gt.shape[-2], # cx
      gtbox[:,-3]*y_pred_gt.shape[-3], # cy
      # shrunk box by self.shr_co 
      gtbox[:,-2]*y_pred_gt.shape[-2]*self.shr_co, # w
      gtbox[:,-1]*y_pred_gt.shape[-3]*self.shr_co], # h
      axis=1)
    csxs = tf.cast(tf.math.floor(sh_gtbox[:,-4]),tf.int32)
    cexs = tf.cast(tf.math.ceil(sh_gtbox[:,-4]+1),tf.int32)
    csys = tf.cast(tf.math.floor(sh_gtbox[:,-3]),tf.int32)
    ceys = tf.cast(tf.math.ceil(sh_gtbox[:,-3]+1),tf.int32)

    # box loss
    label = np.zeros(y_pred_gt.shape[-3:-1],dtype=np.int32)
    for i in range(gtbox.shape[0]):
      label[csys[i]:ceys[i],csxs[i]:cexs[i]] = 1
      bxloss += tf.reduce_mean(
        tf.reduce_sum(tf.math.abs(y_pred_gt[0,csys[i]:ceys[i],csxs[i]:cexs[i],:]-gtbox[i,-4:]),axis=-1)
        )
    bxloss = tf.math.divide(bxloss,gtbox.shape[0])

    return tf.cast(bxloss,tf.float32)

  def call(self, y_true, y_pred):
    ytmask = tf.image.resize(y_true['mask'],y_pred['scr'].shape[-3:-1],'nearest')
    mskloss = self.mask_loss(ytmask,y_pred['mask'],y_true['gt'][:,-4:])
    scloss = self.src_loss(y_ture_mask=ytmask, y_pred_scr=y_pred['scr'])
    bxloss = self.box_loss(gtbox=y_true['gt'][:,-4:],y_pred_gt=y_pred['gt'])
    self.cur_loss={
      'mask':mskloss,
      'box':bxloss,
      'score':scloss,
      }
    return self.ls_fun(bxloss,mskloss,scloss)


class Unet_dbg(tf.keras.Model):
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
    super(Unet_dbg, self).__init__()
    self.std = bool(std)
    self.feat = feat.lower() if(feat.lower() in ['vgg16','resnet'])else 'vgg16'
    self.src_chs = cfg['SRC_CHS']
    self.f_layers_name = cfg['F_LAYER'][self.feat]
    self.fc6 = None
    self.fc6_cls = None
    self.fc7 = None
    self.fc7_cls = None
    self.fc_chs = 512
  
  def build(self,input_shape):
    self.vgg_b1_c1 = tf.keras.layers.Conv2D(
        filters = 64,
        kernel_size=(3,3),
        activation=tf.nn.relu,
        name="vgg_b1_c1",
      )
    self.vgg_b1_c2 = tf.keras.layers.Conv2D(
        filters = 64,
        kernel_size = (3,3),
        activation = tf.nn.relu,
        name="vgg_b1_c2",
      )
    self.vgg_b1_mp = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
      strides=(1, 1), padding='valid')

    self.vgg_b2_c1 = tf.keras.layers.Conv2D(
        filters = 128,
        kernel_size=(3,3),
        activation=tf.nn.relu,
        name="vgg_b2_c1",
      )
    self.vgg_b2_c2 = tf.keras.layers.Conv2D(
        filters = 128,
        kernel_size = (3,3),
        activation = tf.nn.relu,
        name="vgg_b2_c2",
      )
    self.vgg_b2_mp = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
      strides=(1, 1), padding='valid')
  
    self.vgg_b3_c1 = tf.keras.layers.Conv2D(
        filters = 256,
        kernel_size=(3,3),
        activation=tf.nn.relu,
        name="vgg_b3_c1",
      )
    self.vgg_b3_c2 = tf.keras.layers.Conv2D(
        filters = 256,
        kernel_size = (3,3),
        activation = tf.nn.relu,
        name="vgg_b3_c2",
      )
    self.vgg_b3_c3 = tf.keras.layers.Conv2D(
        filters = 256,
        kernel_size = (3,3),
        activation = tf.nn.relu,
        name="vgg_b3_c3",
      )
    self.vgg_b3_mp = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
      strides=(1, 1), padding='valid')

    self.vgg_b4_c1 = tf.keras.layers.Conv2D(
        filters = 512,
        kernel_size=(3,3),
        activation=tf.nn.relu,
        name="vgg_b4_c1",
      )
    self.vgg_b4_c2 = tf.keras.layers.Conv2D(
        filters = 512,
        kernel_size = (3,3),
        activation = tf.nn.relu,
        name="vgg_b4_c2",
      )
    self.vgg_b4_c3 = tf.keras.layers.Conv2D(
        filters = 512,
        kernel_size = (3,3),
        activation = tf.nn.relu,
        name="vgg_b4_c3",
      )
    self.vgg_b4_mp = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
      strides=(1, 1), padding='valid')

    self.vgg_b5_c1 = tf.keras.layers.Conv2D(
        filters = 512,
        kernel_size=(3,3),
        activation=tf.nn.relu,
        name="vgg_b5_c1",
      )
    self.vgg_b5_c2 = tf.keras.layers.Conv2D(
        filters = 512,
        kernel_size = (3,3),
        activation = tf.nn.relu,
        name="vgg_b5_c2",
      )
    self.vgg_b5_c3 = tf.keras.layers.Conv2D(
        filters = 512,
        kernel_size = (3,3),
        activation = tf.nn.relu,
        name="vgg_b5_c3",
      )
    self.vgg_b5_mp = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
      strides=(1, 1), padding='valid')

    self.vgg_fc6 = tf.keras.layers.Conv2D(
        filters = 512,
        kernel_size = (1,1),
        activation = tf.nn.relu,
        name="vgg_fc6",
      )

    self.vgg_fc7 = tf.keras.layers.Conv2D(
        filters = 512,
        kernel_size = (1,1),
        activation = tf.nn.relu,
        name="vgg_fc7",
      )

    self.max_scale_factor = cfg['F_MXS'][self.feat]
    f_layer_num = len(self.feature_model.outputs)

    self.scor_map_list = [
      tf.keras.layers.Conv2D(
        filters = self.src_chs,
        kernel_size=(1, 1),
        activation = cfg['SRC_ACT'],
        name="l{}_score_map".format(i),
      )
      for i in range(f_layer_num)]

    self.rect_list = [
      tf.keras.layers.Conv2D(
        filters = self.feature_model.outputs[i].shape[-1],
        kernel_size=(1, 1),
        activation=tf.nn.relu,
        name="{}_to_{}_1x1_conv".format(i+1,i),
      )
      for i in range(f_layer_num-2,-1,-1)]

    if('fc6' in self.f_layers_name):
      self.fc6 = tf.keras.layers.Conv2D(filters = self.fc_chs, kernel_size=(1, 1), name='fc6')
      self.scor_map_list.insert(0,
        tf.keras.layers.Conv2D(
          filters = self.src_chs,
          kernel_size=(1, 1),
          activation = cfg['SRC_ACT'],
          name="fc6_score_map",
        ))
      self.rect_list.insert(0,tf.keras.layers.Conv2D(
        filters = self.feature_model.outputs[-1].shape[-1],
        kernel_size=(1, 1),
        activation=tf.nn.relu,
        name="fc6_to_{}_1x1_conv".format(f_layer_num-1),
      ))
    if('fc7' in self.f_layers_name):
      if(not('fc6' in self.f_layers_name)):
        self.fc6 = tf.keras.layers.Conv2D(filters = self.fc_chs, kernel_size=(1, 1), name='fc6')
        self.rect_list.insert(0,tf.keras.layers.Conv2D(
          filters = self.feature_model.outputs[-1].shape[-1],
          kernel_size=(1, 1),
          activation=tf.nn.relu,
          name="fc6_to_{}_1x1_conv".format(f_layer_num-1),
        ))
        self.scor_map_list.insert(0,None)  

      self.fc7 = tf.keras.layers.Conv2D(filters = self.fc_chs, kernel_size=(1, 1), name='fc7')
      self.scor_map_list.insert(0,
        tf.keras.layers.Conv2D(
          filters = self.src_chs,
          kernel_size=(1, 1),
          activation = cfg['SRC_ACT'],
          name="fc7_score_map",
        ))
      self.rect_list.insert(0,
        tf.keras.layers.Conv2D(
          filters = self.fc_chs,
          kernel_size=(1, 1),
          activation=tf.nn.relu,
          name="fc7_to_fc6_1x1_conv",
        )
      )
    
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
    super(Unet_dbg, self).build(input_shape)

  def call(self, inputs):
    inputs = tf.image.resize(
      inputs,
      [int(inputs.shape[-3]/self.max_scale_factor)*self.max_scale_factor,int(inputs.shape[-2]/self.max_scale_factor)*self.max_scale_factor])

    if(self.std):inputs = tf.math.abs(tf.image.per_image_standardization(inputs))
    ftb1 = self.vgg_b1_c1(inputs)
    ftb1 = self.vgg_b1_c2(ftb1)
    ftb1 = self.vgg_b1_mp(ftb1)

    ftb2 = self.vgg_b2_c1(ftb1)
    ftb2 = self.vgg_b2_c2(ftb2)
    ftb2 = self.vgg_b2_mp(ftb2)

    ftb3 = self.vgg_b3_c1(ftb2)
    ftb3 = self.vgg_b3_c2(ftb3)
    ftb3 = self.vgg_b3_c3(ftb3)
    ftb3 = self.vgg_b3_mp(ftb3)

    ftb4 = self.vgg_b4_c1(ftb3)
    ftb4 = self.vgg_b4_c2(ftb4)
    ftb4 = self.vgg_b4_c3(ftb4)
    ftb4 = self.vgg_b4_mp(ftb4)

    ftb5 = self.vgg_b5_c1(ftb4)
    ftb5 = self.vgg_b5_c2(ftb5)
    ftb5 = self.vgg_b5_c3(ftb5)
    ftb5 = self.vgg_b5_mp(ftb5)

    ftb6 = self.vgg_fc6(ftb5)
    ftb7 = self.vgg_fc7(ftb6)

    

    mask = self.map_conv(ftb7)
    gts = self.box_conv(ftb7)

    self.ft = ftb7
    # return {'mask':mask,'gt':gts,'scr':scr}
    return {'mask':mask,'gt':gts}