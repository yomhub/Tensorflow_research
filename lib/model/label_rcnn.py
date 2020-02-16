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
  def __init__(self,
    num_classes=2,
    feature_layer_name='vgg16',
    bx_ths=0.7,
    direction=8,
    max_outputs_num=2000,
    neg_rect=1.0,
    pst_rect=1.0,
    ):
    super(Label_RCNN, self).__init__()

    if(feature_layer_name.lower()=='resnet'):
      self.feature_layer_name='resnet'
    else:
      self.feature_layer_name='vgg16'
      # final chs is 512
    self.num_classes = int(num_classes)
    self.bx_ths = bx_ths
    self.direction = direction
    self.neg_rect = float(-neg_rect)
    self.pst_rect = float(pst_rect)

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
      self.rpn_L1_wshape = [1,1,self.feature_model.output[0].shape[-1]]
      self.rpn_L2_wshape = [1,1,self.feature_model.output[1].shape[-1]]
      self.rpn_L3_wshape = [1,1,self.feature_model.output[2].shape[-1]]
      self.rpn_L1_rec_fild = [float(imgh/self.feature_model.output[0].shape[-3]),float(imgw/self.feature_model.output[0].shape[-2])]
      self.rpn_L2_rec_fild = [float(imgh/self.feature_model.output[1].shape[-3]),float(imgw/self.feature_model.output[1].shape[-2])]
      self.rpn_L3_rec_fild = [float(imgh/self.feature_model.output[2].shape[-3]),float(imgw/self.feature_model.output[2].shape[-2])]
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
                            padding="same",
                            # kernel_initializer=unf_pn1,
                            )
    self.rpn_L1_cls_score = tf.keras.layers.Conv2D(filters=self.num_classes,
                            kernel_size=(1, 1),
                            activation=None,
                            padding='same', 
                            name="rpn_L1_cls_score",
                            # kernel_initializer=unf_pn1,
                            )
    self.rpn_L1_bbox_pred = tf.keras.layers.Conv2D(filters=(self.num_classes-1)*4,
                            kernel_size=(1, 1),
                            padding='same', 
                            activation=None,
                            name="rpn_L1_bbox_pred",
                            # kernel_initializer=unf_pn1,
                            )     
    cube_h,cube_w = self.feature_model.output[0].shape[-3],self.feature_model.output[0].shape[-2]
    self.rpn_l1_det = feat_layer_cod_gen([imgh,imgw],[cube_h,cube_w],self.num_classes-1)

    # self.rpn_L1_window = tf.keras.models.Sequential(
    #   [
    #     tf.keras.layers.Dense(256,input_shape=self.rpn_L1_wshape,activation=tf.nn.relu),
    #     tf.keras.layers.Dense(self.direction,activation=None),
    #   ],
    #   "rpn_L1_window_direction"
    # )
    self.rpn_L2_conv = tf.keras.layers.Conv2D(filters=self.feature_model.output[1].shape[-1],
                            kernel_size=(3, 3),
                            activation=tf.nn.relu,
                            padding="same",
                            name="rpn_L2_conv",
                            )
    self.rpn_L2_cls_score = tf.keras.layers.Conv2D(filters=self.num_classes,
                            kernel_size=(1, 1),
                            activation=None,
                            padding='same', 
                            name="rpn_L2_cls_score",
                            )
    self.rpn_L2_bbox_pred = tf.keras.layers.Conv2D(filters=(self.num_classes-1)*4,
                            kernel_size=(1, 1),
                            padding='same', 
                            activation=None,
                            name="rpn_L2_bbox_pred",
                            )
    cube_h,cube_w = self.feature_model.output[1].shape[-3],self.feature_model.output[1].shape[-2]
    self.rpn_l2_det = feat_layer_cod_gen([imgh,imgw],[cube_h,cube_w],self.num_classes-1)

    # self.rpn_L2_window = tf.keras.models.Sequential(
    #   [
    #     tf.keras.layers.Dense(512,input_shape=self.rpn_L2_wshape,activation=tf.nn.relu),
    #     tf.keras.layers.Dense(self.direction,activation=None),
    #   ],
    #   "rpn_L2_window_direction"
    # )
    self.rpn_L3_conv = tf.keras.layers.Conv2D(filters=self.feature_model.output[2].shape[-1],
                            kernel_size=(3, 3),
                            activation=tf.nn.relu,
                            padding="same",
                            name="rpn_L3_conv",
                            )
    self.rpn_L3_cls_score = tf.keras.layers.Conv2D(filters=self.num_classes,
                            kernel_size=(1, 1),
                            activation=None,
                            padding='same', 
                            name="rpn_L3_cls_score",
                            )
    self.rpn_L3_bbox_pred = tf.keras.layers.Conv2D(filters=(self.num_classes-1)*4,
                            kernel_size=(1, 1),
                            padding='same', 
                            activation=None,
                            name="rpn_L3_bbox_pred",
                            )     
    cube_h,cube_w = self.feature_model.output[2].shape[-3],self.feature_model.output[2].shape[-2]
    self.rpn_l3_det = feat_layer_cod_gen([imgh,imgw],[cube_h,cube_w],self.num_classes-1)                          
    # self.rpn_L3_window = tf.keras.models.Sequential(
    #   [
    #     tf.keras.layers.Dense(512,input_shape=self.rpn_L3_wshape,activation=tf.nn.relu),
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

    l1_bbox = tf.stack([
      # limit det y1,x1 in [0,0.5~0.6] because we use 
      # round down GT y1,x1, so it only had positive det
      tf.clip_by_value(l1_bbox[:,:,:,0],self.neg_rect,self.pst_rect)*self.rpn_L1_rec_fild[0],
      tf.clip_by_value(l1_bbox[:,:,:,1],self.neg_rect,self.pst_rect)*self.rpn_L1_rec_fild[1],
      # Use round up GT y2,x2, so it only had negative det
      tf.clip_by_value(l1_bbox[:,:,:,2],self.neg_rect,self.pst_rect)*self.rpn_L1_rec_fild[0],
      tf.clip_by_value(l1_bbox[:,:,:,3],self.neg_rect,self.pst_rect)*self.rpn_L1_rec_fild[1],
      ],
      axis=-1)
    # l1_ort = tf.nn.softmax(self.rpn_L1_window(l1_feat),axis=-1)
    l2_bbox = tf.stack([
      tf.clip_by_value(l2_bbox[:,:,:,0],self.neg_rect,self.pst_rect)*self.rpn_L2_rec_fild[0],
      tf.clip_by_value(l2_bbox[:,:,:,1],self.neg_rect,self.pst_rect)*self.rpn_L2_rec_fild[1],
      tf.clip_by_value(l2_bbox[:,:,:,2],self.neg_rect,self.pst_rect)*self.rpn_L2_rec_fild[0],
      tf.clip_by_value(l2_bbox[:,:,:,3],self.neg_rect,self.pst_rect)*self.rpn_L2_rec_fild[1],
      ],
      axis=-1)
    # l2_ort = tf.nn.softmax(self.rpn_L2_window(l2_feat),axis=-1)
    l3_bbox = tf.stack([
      tf.clip_by_value(l3_bbox[:,:,:,0],self.neg_rect,self.pst_rect)*self.rpn_L3_rec_fild[0],
      tf.clip_by_value(l3_bbox[:,:,:,1],self.neg_rect,self.pst_rect)*self.rpn_L3_rec_fild[1],
      tf.clip_by_value(l3_bbox[:,:,:,2],self.neg_rect,self.pst_rect)*self.rpn_L3_rec_fild[0],
      tf.clip_by_value(l3_bbox[:,:,:,3],self.neg_rect,self.pst_rect)*self.rpn_L3_rec_fild[1],
      ],
      axis=-1)

    l1_bbox += self.rpn_l1_det
    l2_bbox += self.rpn_l2_det
    l3_bbox += self.rpn_l3_det
    # l3_ort = tf.nn.softmax(self.rpn_L3_window(l3_feat),axis=-1)

    # 
    self.y_pred = {
      "l1_score" : l1_score,
      "l1_bbox" : l1_bbox,
      # "l1_ort" : l1_ort,
      "l2_score" : l2_score,
      "l2_bbox" : l2_bbox,
      # "l2_ort" : l2_ort,
      "l3_score" : l3_score,
      "l3_bbox" : l3_bbox,
      # "l3_ort" : l3_ort,
    }
    # return l1_score,l1_bbox,l1_ort,l2_score,l2_bbox,l2_ort,l3_score,l3_bbox,l3_ort
    return self.y_pred

class LRCNNLoss(tf.keras.losses.Loss):
  def __init__(self,imge_size,gtformat='yxyx'):
    super(LRCNNLoss, self).__init__()
    gtformat = gtformat.lower()
    if(gtformat.lower()=='yxyx' or gtformat.lower()=='yx'):
      self.gtformat='yxyx'
    else:
      self.gtformat='xywh'
    self.imge_size = imge_size

  def _label_loss(self,pred_score,y_true):
    labels = gen_label_from_gt(pred_score.shape[1:3],y_true,self.imge_size,0)
    labels = tf.reshape(labels,[-1])
    # select = tf.reshape(tf.where(tf.not_equal(labels, -1)),[-1])
    select1 = tf.reshape(tf.where(tf.equal(labels, 1)),[-1])
    select0 = tf.reshape(tf.where(tf.equal(labels, 0)),[-1])
    # score = tf.gather(tf.reshape(pred_score,[-1,2]), select)
    score = tf.concat([
      tf.gather(tf.reshape(pred_score[:,1],[-1]), select1),
      tf.gather(tf.reshape(pred_score[:,0],[-1]), select0),
      ],axis=0)
    labels = tf.cast(tf.concat([
      tf.gather(labels, select1),
      tf.gather(labels, select0),
      ],axis=0),tf.float32)
    # labels = tf.gather(labels,select)
    # score = tf.nn.softmax(tf.gather(score, select),axis=-1)
    # score = tf.reshape(score[:,1],[1,select.shape[0]])
    # labels = tf.reshape(tf.gather(labels,select),[1,select.shape[0]])
    # loss = tf.keras.losses.binary_crossentropy(labels,score,from_logits=False)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=score, labels=labels)
    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score, labels=labels))
    return tf.reduce_mean(loss)
    
  def _boxes_loss(self,box_prd,y_true):
    loss_value = pre_box_loss(y_true,box_prd,self.imge_size)
    # loss_value = loss_value / tf.math.reduce_sum((y_true[:,2]-y_true[:,0])*(y_true[:,3]-y_true[:,1]))
    return tf.math.reduce_mean(loss_value)

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

    # gt_area = gt_area_h*gt_area_w
    # l3_y = tf.gather(y_true,tf.reshape(tf.where(gt_area>=32.0*32.0),[-1]))
    # if(l3_y.shape[0]!=None and l3_y.shape[0]>0):
    #   l3_box_loss = self._boxes_loss(y_pred["l3_bbox"],l3_y[:,-4:])
    # else:
    #   l3_box_loss = 0.0
    l3_box_loss = self._boxes_loss(y_pred["l3_bbox"],y_true[:,-4:])
    l3_label_loss = self._label_loss(y_pred["l3_score"],y_true)

    # l2_y = tf.gather(y_true,tf.reshape(tf.where(tf.logical_and(gt_area>=16.0*16.0,gt_area<32.0*32.0)),[-1]))
    # if(l2_y.shape[0]!=None and l2_y.shape[0]>0):
    #   l2_box_loss = self._boxes_loss(y_pred["l2_bbox"],l2_y[:,-4:])
    # else:
    #   l2_box_loss = 0.0
    l2_box_loss = self._boxes_loss(y_pred["l2_bbox"],y_true[:,-4:])
    l2_label_loss = self._label_loss(y_pred["l2_score"],y_true)

    # l1_y = tf.gather(y_true,tf.reshape(tf.where(gt_area<16.0*16.0),[-1]))
    # if(l1_y.shape[0]!=None and l1_y.shape[0]>0):
    #   l1_box_loss = self._boxes_loss(y_pred["l1_bbox"],l1_y[:,-4:])
    # else:
    #   l1_box_loss = 0.0
    l1_box_loss = self._boxes_loss(y_pred["l1_bbox"],y_true[:,-4:])
    l1_label_loss = self._label_loss(y_pred["l1_score"],y_true)

    self.loss_detail={
      "l1_label_loss" : l1_label_loss,
      "l2_label_loss" : l2_label_loss,
      "l3_label_loss" : l3_label_loss,
      "l1_box_loss" : l1_box_loss,
      "l2_box_loss" : l2_box_loss,
      "l3_box_loss" : l3_box_loss,
    }

    return l1_label_loss + l2_label_loss + l3_label_loss + l1_box_loss + l2_box_loss + l3_box_loss