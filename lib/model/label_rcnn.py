from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
import math
from model.config import cfg
from tflib.bbox_transform import *
from tflib.anchor_target_layer import anchor_target_layer_tf, point_anchor_target_layer_tf
from tflib.proposal_target_layer import proposal_target_layer_tf
from tflib.snippets import generate_real_anchors, score_convert
from tflib.common import *


class Label_RCNN(tf.keras.Model):
  """
    Args:
      direction: int of predict box relation in 8 or 4 direction
      mod: prediction model in 'yxyx' or 
        final prediction will be:
        'yxyx': [dy1*win_h,dx1*win_w,dy2*win_h,dx2*win_w] 
        'yxhw': [dys,dxs,dh,dw]
  """
  def __init__(self,
    num_classes=2,
    feature_layer_name='vgg16',
    direction=8,
    mod='yxyx'
    ):
    super(Label_RCNN, self).__init__()
    self.feature_layer_name = feature_layer_name.lower()
    if(not(self.feature_layer_name in ['resnet','vgg16'])):
      self.feature_layer_name='vgg16'
    self.num_classes = int(num_classes)
    self.direction = 4 if direction<=4 else 8
    self.mod = 'yxyx' if(mod=='yxyx')else 'yxhw'

  def build(self,input_shape):
    imgh = input_shape[-3]
    imgw = input_shape[-2]
    if(self.feature_layer_name=='vgg16'):
      vgg16=tf.keras.applications.VGG16(
        input_tensor=tf.keras.Input(shape=input_shape[-3:]),
        weights='imagenet', 
        include_top=False)
      self.feature_model = tf.keras.Model(
        inputs=vgg16.inputs,
        outputs=[
          # vgg16.get_layer('block2_conv2').output,
          # 2*2 128
          # vgg16.get_layer('block3_conv3').output,
          # 4*4 256
          vgg16.get_layer('block3_pool').output,
          # 8*8 256
          # vgg16.get_layer('block4_conv3').output,
          # 8*8 512
          vgg16.get_layer('block4_pool').output,
          # 16*16 512
          # vgg16.get_layer('block5_conv3').output
          # 16*16 512
          vgg16.get_layer('block5_pool').output
          # 32*32 512
        ],
        name=self.feature_layer_name
      )
    elif(self.feature_layer_name=='resnet'):
      rn=tf.keras.applications.resnet_v2.ResNet101V2()
        # rn.get_layer("conv1_pad"), rn.get_layer("conv1_conv"), rn.get_layer("pool1_pad"), rn.get_layer("pool1_pool"),
        # rn.get_layer("block2_conv1"), rn.get_layer("block2_conv2"), rn.get_layer("block2_pool"),
        # rn.get_layer("block3_conv1"), rn.get_layer("block3_conv2"), rn.get_layer("block3_conv3"), 
        # rn.get_layer("block3_pool"),
        # rn.get_layer("block4_conv1"), rn.get_layer("block4_conv2"), rn.get_layer("block4_conv3"),
        # rn.get_layer("block4_pool"),
        # rn.get_layer("block5_conv1"), rn.get_layer("block5_conv2"), rn.get_layer("block5_conv3"),
        # rn.get_layer("block5_pool"),
      self.feature_model = tf.keras.Model(
        inputs=rn.input,
        outputs=rn.get_layer("block5_pool").output,
        name=self.feature_layer_name
      )
    # unf_pn1 = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
    self.rpn_L1_conv = tf.keras.layers.Conv2D(filters=self.feature_model.output[0].shape[-1],
                            kernel_size=(3, 3),
                            activation=tf.nn.relu,
                            name="rpn_L1_conv",
                            # padding="same",
                            # kernel_initializer=unf_pn1,
                            )
    self.rpn_L1_cls_score = tf.keras.layers.Conv2D(filters=self.num_classes,
                            kernel_size=(1, 1),
                            activation=None,
                            # padding='same', 
                            name="rpn_L1_cls_score",
                            # kernel_initializer=unf_pn1,
                            )
    self.rpn_L1_bbox_pred = tf.keras.layers.Conv2D(filters=(self.num_classes-1)*4,
                            kernel_size=(1, 1),
                            # padding='same', 
                            activation=None,
                            name="rpn_L1_bbox_pred",
                            # kernel_initializer=unf_pn1,
                            )
    cube_h,cube_w = self.rpn_L1_conv(self.feature_model.output[0]).shape[-3],self.rpn_L1_conv(self.feature_model.output[0]).shape[-2]
    self.rpn_L1_rec_fild = [float(imgh/cube_h),float(imgw/cube_w)]
    self.rpn_L1_rec_fild = tf.convert_to_tensor([self.rpn_L1_rec_fild[0],self.rpn_L1_rec_fild[1],self.rpn_L1_rec_fild[0],self.rpn_L1_rec_fild[1]],dtype=tf.float32)
    self.rpn_l1_det = feat_layer_cod_gen([imgh,imgw],[cube_h,cube_w],self.num_classes-1)

    # self.rpn_L1_window = tf.keras.models.Sequential(
    #   [
    #     tf.keras.layers.Dense(256,input_shape=[1,1,self.feature_model.output[0].shape[-1]],activation=tf.nn.relu),
    #     tf.keras.layers.Dense(self.direction,activation=None),
    #   ],
    #   "rpn_L1_window_direction"
    # )
    self.rpn_L2_conv = tf.keras.layers.Conv2D(filters=self.feature_model.output[1].shape[-1],
                            kernel_size=(3, 3),
                            activation=tf.nn.relu,
                            # padding="same",
                            name="rpn_L2_conv",
                            )
    self.rpn_L2_cls_score = tf.keras.layers.Conv2D(filters=self.num_classes,
                            kernel_size=(1, 1),
                            activation=None,
                            # padding='same', 
                            name="rpn_L2_cls_score",
                            )
    self.rpn_L2_bbox_pred = tf.keras.layers.Conv2D(filters=(self.num_classes-1)*4,
                            kernel_size=(1, 1),
                            # padding='same', 
                            activation=None,
                            name="rpn_L2_bbox_pred",
                            )
    cube_h,cube_w = self.rpn_L2_conv(self.feature_model.output[1]).shape[-3],self.rpn_L2_conv(self.feature_model.output[1]).shape[-2]
    self.rpn_L2_rec_fild = [float(imgh/cube_h),float(imgw/cube_w)]
    self.rpn_L2_rec_fild = tf.convert_to_tensor([self.rpn_L2_rec_fild[0],self.rpn_L2_rec_fild[1],self.rpn_L2_rec_fild[0],self.rpn_L2_rec_fild[1]],dtype=tf.float32)
    self.rpn_l2_det = feat_layer_cod_gen([imgh,imgw],[cube_h,cube_w],self.num_classes-1)

    # self.rpn_L2_window = tf.keras.models.Sequential(
    #   [
    #     tf.keras.layers.Dense(512,input_shape=[1,1,self.feature_model.output[1].shape[-1]],activation=tf.nn.relu),
    #     tf.keras.layers.Dense(self.direction,activation=None),
    #   ],
    #   "rpn_L2_window_direction"
    # )
    self.rpn_L3_conv = tf.keras.layers.Conv2D(filters=self.feature_model.output[2].shape[-1],
                            kernel_size=(3, 3),
                            activation=tf.nn.relu,
                            # padding="same",
                            name="rpn_L3_conv",
                            )
    self.rpn_L3_cls_score = tf.keras.layers.Conv2D(filters=self.num_classes,
                            kernel_size=(1, 1),
                            activation=None,
                            # padding='same', 
                            name="rpn_L3_cls_score",
                            )
    self.rpn_L3_bbox_pred = tf.keras.layers.Conv2D(filters=(self.num_classes-1)*4,
                            kernel_size=(1, 1),
                            # padding='valid', 
                            # padding='same', 
                            activation=None,
                            name="rpn_L3_bbox_pred",
                            )
    cube_h,cube_w = self.rpn_L3_conv(self.feature_model.output[2]).shape[-3],self.rpn_L3_conv(self.feature_model.output[2]).shape[-2]
    self.rpn_L3_rec_fild = [float(imgh/cube_h),float(imgw/cube_w)]
    self.rpn_L3_rec_fild = tf.convert_to_tensor([self.rpn_L3_rec_fild[0],self.rpn_L3_rec_fild[1],self.rpn_L3_rec_fild[0],self.rpn_L3_rec_fild[1]],dtype=tf.float32)
    self.rpn_l3_det = feat_layer_cod_gen([imgh,imgw],[cube_h,cube_w],self.num_classes-1)                          
    # self.rpn_L3_window = tf.keras.models.Sequential(
    #   [
    #     tf.keras.layers.Dense(512,input_shape=[1,1,self.feature_model.output[2].shape[-1]],activation=tf.nn.relu),
    #     tf.keras.layers.Dense(self.direction,activation=None),
    #   ],
    #   "rpn_L3_window_direction"
    # )         
    self.imgh = imgh
    self.imgw = imgw
    super(Label_RCNN, self).build(input_shape)

  def call(self, inputs):
    if(inputs.dtype!=tf.float32 or inputs.dtype!=tf.float64):
      inputs = tf.cast(inputs,tf.float32)
    if(inputs.shape[-3]!=self.imgh or inputs.shape[-2]!=self.imgw):
      inputs = tf.image.resize(inputs,[self.imgh,self.imgw])
    if(len(inputs.shape)==3):
      inputs = tf.reshape(inputs,[1]+inputs.shape)
    l1_feat,l2_feat,l3_feat = self.feature_model(inputs)
    l1_feat = self.rpn_L1_conv(l1_feat)
    l2_feat = self.rpn_L2_conv(l2_feat)
    l3_feat = self.rpn_L3_conv(l3_feat)

    # limit every bbox difference value in [-1,1] 
    l1_score = self.rpn_L1_cls_score(l1_feat)
    l2_score = self.rpn_L2_cls_score(l2_feat)
    l3_score = self.rpn_L3_cls_score(l3_feat)

    l1_bbox = self.rpn_L1_bbox_pred(l1_feat)
    l2_bbox = self.rpn_L2_bbox_pred(l2_feat)
    l3_bbox = self.rpn_L3_bbox_pred(l3_feat)

    # bbox_cod in [y1,x1,y2,x2]
    if(self.mod=='yxyx'):
      l1_bbox_cod = l1_bbox * self.rpn_L1_rec_fild + self.rpn_l1_det
      l2_bbox_cod = l2_bbox * self.rpn_L2_rec_fild + self.rpn_l2_det
      l3_bbox_cod = l3_bbox * self.rpn_L3_rec_fild + self.rpn_l3_det
    else:
      l1_bbox_cod = tf.stack([
        l1_bbox[:,:,:,0]*self.rpn_L1_rec_fild[0] + self.rpn_l1_det[:,:,:,0],
        l1_bbox[:,:,:,1]*self.rpn_L1_rec_fild[1] + self.rpn_l1_det[:,:,:,1],
        tf.math.exp(l1_bbox[:,:,:,2])*self.rpn_L1_rec_fild[0] + l1_bbox[:,:,:,0]*self.rpn_L1_rec_fild[0] + self.rpn_l1_det[:,:,:,0],
        tf.math.exp(l1_bbox[:,:,:,3])*self.rpn_L1_rec_fild[1] + l1_bbox[:,:,:,1]*self.rpn_L1_rec_fild[1] + self.rpn_l1_det[:,:,:,1],
        ],axis=1)
      l2_bbox_cod = tf.stack([
        l2_bbox[:,:,:,0]*self.rpn_L2_rec_fild[0] + self.rpn_l2_det[:,:,:,0],
        l2_bbox[:,:,:,1]*self.rpn_L2_rec_fild[1] + self.rpn_l2_det[:,:,:,1],
        tf.math.exp(l2_bbox[:,:,:,2])*self.rpn_L2_rec_fild[0] + l2_bbox[:,:,:,0]*self.rpn_L2_rec_fild[0] + self.rpn_l2_det[:,:,:,0],
        tf.math.exp(l2_bbox[:,:,:,3])*self.rpn_L2_rec_fild[1] + l2_bbox[:,:,:,1]*self.rpn_L2_rec_fild[1] + self.rpn_l2_det[:,:,:,1],
        ],axis=1)
      l3_bbox_cod =tf.stack([
        l2_bbox[:,:,:,0]*self.rpn_L2_rec_fild[0] + self.rpn_l2_det[:,:,:,0],
        l2_bbox[:,:,:,1]*self.rpn_L2_rec_fild[1] + self.rpn_l2_det[:,:,:,1],
        tf.math.exp(l2_bbox[:,:,:,2])*self.rpn_L2_rec_fild[0] + l2_bbox[:,:,:,0]*self.rpn_L2_rec_fild[0] + self.rpn_l2_det[:,:,:,0],
        tf.math.exp(l2_bbox[:,:,:,3])*self.rpn_L2_rec_fild[1] + l2_bbox[:,:,:,1]*self.rpn_L2_rec_fild[1] + self.rpn_l2_det[:,:,:,1],
        ],axis=1)
    # l1_ort = tf.nn.softmax(self.rpn_L1_window(l1_feat),axis=-1)
    # l2_ort = tf.nn.softmax(self.rpn_L2_window(l2_feat),axis=-1)
    # l3_ort = tf.nn.softmax(self.rpn_L3_window(l3_feat),axis=-1)

    self.y_pred = {
      "l1_score" : l1_score,
      "l1_bbox" : l1_bbox_cod,
      "l1_bbox_det" : l1_bbox,
      # "l1_ort" : l1_ort,
      "l2_score" : l2_score,
      "l2_bbox" : l2_bbox_cod,
      "l2_bbox_det" : l2_bbox,
      # "l2_ort" : l2_ort,
      "l3_score" : l3_score,
      "l3_bbox" : l3_bbox_cod,
      "l3_bbox_det" : l3_bbox,
      # "l3_ort" : l3_ort,
    }
    # return l1_score,l1_bbox,l1_ort,l2_score,l2_bbox,l2_ort,l3_score,l3_bbox,l3_ort
    return self.y_pred

class LRCNNLoss(tf.keras.losses.Loss):
  def __init__(self,imge_size,gtformat='yxyx',use_cross=True,mag_f='smooth'):
    """
      Args:
        use_cross: set True to use cross loss function
        mag_f: magnification function, can be
          string: 'smooth', apply L1 smooth on loss function 
          string: 'sigmoid', apply sigmoid on loss function 
          tf.keras.layers.Lambda: apply input Lambda object
          others, don't apply any magnification function
    """
    super(LRCNNLoss, self).__init__()
    self.gtformat = gtformat.lower()
    if(not(self.gtformat in ['yxyx','xywh','mask'])):
      self.gtformat='xywh'
    self.imge_size = imge_size
    self.use_cross = use_cross
    self.mag_f = mag_f

  def _label_loss(self,pred_score,y_true):
    # sigmoid get a divergent loss
    losstype='softmax'
    labels, weights = gen_label_with_width_from_gt(pred_score.shape[1:3],y_true,self.imge_size,0)
    labels = tf.reshape(labels,[-1])
    weights = tf.reshape(weights,[-1])
    if(losstype=='sigmoid'):
      select1 = tf.reshape(tf.where(tf.equal(labels, 1)),[-1])
      select0 = tf.reshape(tf.where(tf.equal(labels, 0)),[-1])
      score = tf.concat([
        tf.gather(tf.reshape(pred_score[:,1],[-1]), select1),
        # tf.gather(tf.reshape(pred_score[:,0],[-1]), select0),
        ],axis=0)
      labels = tf.cast(tf.concat([
        tf.gather(labels, select1),
        # tf.gather(labels, select0),
        ],axis=0),tf.float32)
      weights = tf.cast(tf.concat([
        tf.gather(weights, select1),
        # tf.gather(weights, select0),
        ],axis=0),tf.float32)
    # loss = tf.keras.losses.binary_crossentropy(labels,score,from_logits=False)
      loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=score, labels=labels)
    else:
      select = tf.reshape(tf.where(tf.not_equal(labels, -1)),[-1])
      score = tf.gather(tf.reshape(pred_score,[-1,pred_score.shape[-1]]), select)
      labels = tf.gather(labels,select)
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score, labels=labels)
    return tf.reduce_mean(loss)

  def _boxes_loss(self,box_prd,y_true):

    loss_value = pre_box_loss_by_det(y_true,box_prd,self.imge_size,use_cross=self.use_cross,mag_f=self.mag_f)
    # loss_value = pre_box_loss(y_true,box_prd,self.imge_size)
    # loss_value = loss_value / tf.math.reduce_sum((y_true[:,2]-y_true[:,0])*(y_true[:,3]-y_true[:,1]))
    loss_value = tf.math.reduce_mean(loss_value)

    return loss_value

  def _boxes_label_loss(self,box_prd,pred_score,y_true):
    # ONLY for mask gt
    box_loss,label_loss = pre_box_loss_by_msk(y_true,box_prd,pred_score,self.imge_size,use_pixel=False)
    return box_loss,label_loss

  def call(self, y_true, y_pred):
    if(self.gtformat=='xywh'):
      # convert [class, xstart, ystart, w, h] to
      # [y1, x1, y2, x2]
      gt_area_h = tf.reshape(y_true[:,-1],[-1])
      gt_area_w = tf.reshape(y_true[:,-2],[-1])
      gt_boxes = xywh2yxyx(y_true[:,1:])
      y_true = tf.stack([y_true[:,0],gt_boxes[:,0],gt_boxes[:,1],gt_boxes[:,2],gt_boxes[:,3]],axis=1)
    else:
      gt_area_h = tf.reshape(y_true[:,-2]-y_true[:,-4],[-1])
      gt_area_w = tf.reshape(y_true[:,-1]-y_true[:,-3],[-1])

    if(self.gtformat!='mask'):
      l3_box_loss = self._boxes_loss(y_pred["l3_bbox_det"],y_true[:,-4:])
      l2_box_loss = self._boxes_loss(y_pred["l2_bbox_det"],y_true[:,-4:])
      l1_box_loss = self._boxes_loss(y_pred["l1_bbox_det"],y_true[:,-4:])
      l3_label_loss = self._label_loss(y_pred["l3_score"],y_true)
      l2_label_loss = self._label_loss(y_pred["l2_score"],y_true)
      l1_label_loss = self._label_loss(y_pred["l1_score"],y_true)
    else:
      l3_box_loss,l3_label_loss = self._boxes_label_loss(y_pred["l3_bbox_det"],y_pred["l3_score"],y_true)
      l2_box_loss,l2_label_loss = self._boxes_label_loss(y_pred["l2_bbox_det"],y_pred["l2_score"],y_true)
      l1_box_loss,l1_label_loss = self._boxes_label_loss(y_pred["l1_bbox_det"],y_pred["l1_score"],y_true)

    self.loss_detail={
      "l1_label_loss" : l1_label_loss,
      "l2_label_loss" : l2_label_loss,
      "l3_label_loss" : l3_label_loss,
      "l1_box_loss" : l1_box_loss,
      "l2_box_loss" : l2_box_loss,
      "l3_box_loss" : l3_box_loss,
    }

    return l1_box_loss + l2_box_loss + l3_box_loss
    # return l1_label_loss + l2_label_loss + l3_label_loss + l1_box_loss + l2_box_loss + l3_box_loss