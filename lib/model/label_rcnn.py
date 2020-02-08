from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
import math
from model.config import cfg
# from tflib.bbox_transform import bbox_transform_inv_tf, clip_boxes_tf, xywh2yxyx, label_layer, build_boxex_from_path, map2coordinate
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
    nms_thresh=0.6,
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
  def build(self,input_shape):
    
    if(self.feature_layer_name=='vgg16'):
      vgg16=tf.keras.applications.VGG16(weights='imagenet', include_top=False)
      self.feature_model = tf.keras.Model(
        inputs=vgg16.input,
        outputs=[
          # vgg16.get_layer('block2_conv2').output,
          # 2*2 128
          vgg16.get_layer('block3_conv3').output,
          # 4*4 256
          # vgg16.get_layer('block3_pool').output,
          # 8*8 256
          vgg16.get_layer('block4_conv3').output,
          # 8*8 512
          # vgg16.get_layer('block4_pool').output,
          # 16*16 512
          vgg16.get_layer('block5_conv3').output
          # 16*16 512
        ],
        name=self.feature_layer_name
      )
      self.rpn_L1_wshape = [1,1,256]
      self.rpn_L2_wshape = [1,1,512]
      self.rpn_L3_wshape = [1,1,512]
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

    self.rpn_L1_conv = tf.keras.layers.Conv2D(filters=256,
                            kernel_size=(3, 3),
                            activation=tf.nn.relu,
                            name="rpn_L1_conv",
                            padding="same",
                            )
    self.rpn_L1_cls_score = tf.keras.layers.Conv2D(filters=self.num_classes,
                            kernel_size=(1, 1),
                            activation=None,
                            padding='same', 
                            name="rpn_L1_cls_score",
                            )
    self.rpn_L1_bbox_pred = tf.keras.layers.Conv2D(filters=(self.num_classes-1)*4,
                            kernel_size=(1, 1),
                            padding='same', 
                            activation=None,
                            name="rpn_L1_bbox_pred",
                            )                           
    
    self.rpn_L1_window = tf.keras.models.Sequential(
      [
        tf.keras.layers.Dense(256,input_shape=self.rpn_L1_wshape,activation=tf.nn.relu),
        tf.keras.layers.Dense(self.direction,activation=None),
      ],
      "rpn_L1_window_direction"
    )
    self.rpn_L2_conv = tf.keras.layers.Conv2D(filters=512,
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
    self.rpn_L2_window = tf.keras.models.Sequential(
      [
        tf.keras.layers.Dense(512,input_shape=self.rpn_L2_wshape,activation=tf.nn.relu),
        tf.keras.layers.Dense(self.direction,activation=None),
      ],
      "rpn_L2_window_direction"
    )
    self.rpn_L3_conv = tf.keras.layers.Conv2D(filters=512,
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
    self.rpn_L3_window = tf.keras.models.Sequential(
      [
        tf.keras.layers.Dense(512,input_shape=self.rpn_L3_wshape,activation=tf.nn.relu),
        tf.keras.layers.Dense(self.direction,activation=None),
      ],
      "rpn_L3_window_direction"
    )                
    super(Label_RCNN, self).build(input_shape)

  def call(self, inputs):
    if(inputs.dtype!=tf.float32 or inputs.dtype!=tf.float64):
      inputs = tf.cast(inputs,tf.float32)
    l1_feat,l2_feat,l3_feat = self.feature_model(inputs)
    l1_feat = self.rpn_L1_conv(l1_feat)
    l2_feat = self.rpn_L2_conv(l2_feat)
    l3_feat = self.rpn_L3_conv(l3_feat)

    l1_score = self.rpn_L1_cls_score(l1_feat)
    l1_bbox = self.rpn_L1_bbox_pred(l1_feat)
    l1_ort = tf.nn.softmax(self.rpn_L1_window(l1_feat),axis=-1)
    l2_score = self.rpn_L2_cls_score(l2_feat)
    l2_bbox = self.rpn_L2_bbox_pred(l2_feat)
    l2_ort = tf.nn.softmax(self.rpn_L2_window(l2_feat),axis=-1)
    l3_score = self.rpn_L3_cls_score(l3_feat)
    l3_bbox = self.rpn_L3_bbox_pred(l3_feat)
    l3_ort = tf.nn.softmax(self.rpn_L3_window(l3_feat),axis=-1)
  
    self.y_pred = {
      "l1_score" : l1_score,
      "l1_bbox" : l1_bbox,
      "l1_ort" : l1_ort,
      "l2_score" : l2_score,
      "l2_bbox" : l2_bbox,
      "l2_ort" : l2_ort,
      "l3_score" : l3_score,
      "l3_bbox" : l3_bbox,
      "l3_ort" : l3_ort,
    }
    # return l1_score,l1_bbox,l1_ort,l2_score,l2_bbox,l2_ort,l3_score,l3_bbox,l3_ort
    return self.y_pred

class LabelLoss(tf.keras.losses.Loss):
  def __init__(self,imge_size,gtformat='yxyx'):
    super(LabelLoss, self).__init__()
    gtformat = gtformat.lower()
    if(gtformat=='yxyx' or gtformat=='yx'):
      self.gtformat='yxyx'
    else:
      self.gtformat='xywh'
    self.imge_size = imge_size

  def _label_loss(self,pred_score,y_true):
    labels = label_layer(pred_score.shape[1:3],y_true,self.imge_size)
    labels = tf.reshape(labels,[-1])
    select = tf.reshape(tf.where(tf.not_equal(labels, -1)),[-1])
    score = tf.gather(tf.reshape(pred_score,[-1,2]), select)
    labels = tf.gather(labels,select)
    return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score, labels=labels))
    
  def _boxes_loss(self,cls_prb,box_prd,ort_map,y_true):
    path_list, mask_np = build_boxex_from_path(cls_prb,box_prd,ort_map,1)
    y_true = map2coordinate(y_true[1:],self.imge_size,cls_prb.shape[1:3])
    label_list = get_label_from_mask(y_true,mask_np)
    
    return 0

  def call(self, y_true, y_pred):
    if(self.gtformat=='xywh'):
      # convert [class, xstart, ystart, w, h] to
      # [y1, x1, y2, x2]
      gt_boxes = xywh2yxyx(y_true[:,1:])
      y_true = tf.stack([y_true[:,0],gt_boxes[:,0],gt_boxes[:,1],gt_boxes[:,2],gt_boxes[:,3]],axis=1)

    l1_loss = self._label_loss(y_pred["l1_score"],y_true)
    l2_loss = self._label_loss(y_pred["l2_score"],y_true)
    l3_loss = self._label_loss(y_pred["l3_score"],y_true)
    l1_box_loss = self._boxes_loss(y_pred["l1_score"],y_pred["l1_bbox"],y_pred["l1_ort"],y_true)


    return l1_loss + l2_loss + l3_loss