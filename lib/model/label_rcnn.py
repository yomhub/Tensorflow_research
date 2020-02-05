from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
import math
from layer_utils.generate_anchors import generate_anchors
from layer_utils.snippets import generate_anchors_pre, generate_anchors_pre_tf
from model.config import cfg
from tflib.bbox_transform import bbox_transform_inv_tf, clip_boxes_tf, xywh2yxyx, labelLayer
from tflib.anchor_target_layer import anchor_target_layer_tf, point_anchor_target_layer_tf
from tflib.proposal_target_layer import proposal_target_layer_tf
from tflib.snippets import generate_real_anchors, score_convert
from tflib.common import *


class Label_RCNN(tf.keras.Model):
  def __init__(self,
    num_classes=2,
    feature_layer_name='vgg16',
    bx_choose="nms",
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

  def build(self,input_shape):
    if(len(input_shape)==4):
      height=input_shape[1]
      width=input_shape[2]
    else:
      height=input_shape[0]
      width=input_shape[1]

    if(self.feature_layer_name=='vgg16'):
      vgg16=tf.keras.applications.VGG16(weights='imagenet', include_top=False)
      self.feature_model = tf.keras.Model(
        inputs=vgg16.input,
        outputs=[
          vgg16.get_layer('block3_conv3').output,
          # vgg16.get_layer('block3_pool').output,
          vgg16.get_layer('block4_conv3').output,
          # vgg16.get_layer('block4_pool').output,
          vgg16.get_layer('block5_conv3').output
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

    self.rpn_L1_conv = tf.keras.layers.Conv2D(filters=256,
                            kernel_size=(3, 3),
                            activation=tf.nn.relu,
                            name="rpn_L1_conv",
                            padding="same",
                            )
    self.rpn_L1_cls_score = tf.keras.layers.Conv2D(filters=self.num_classes,
                             kernel_size=(1, 1),
                            activation=None,
                            name="rpn_L1_cls_score",
                            )
    self.rpn_L1_bbox_pred = tf.keras.layers.Conv2D(filters=self.num_classes*4,
                            kernel_size=(1, 1),
                            padding='same', 
                            activation=None,
                            name="rpn_L1_bbox_pred",
                            )                           
    self.rpn_L2_conv = tf.keras.layers.Conv2D(filters=512,
                            kernel_size=(3, 3),
                            activation=tf.nn.relu,
                            name="rpn_L2_conv",
                            padding="same",
                            )
    self.rpn_L2_cls_score = tf.keras.layers.Conv2D(filters=self.num_classes,
                            kernel_size=(1, 1),
                            activation=None,
                            name="rpn_L2_cls_score",
                            )
    self.rpn_L2_bbox_pred = tf.keras.layers.Conv2D(filters=self.num_classes*4,
                            kernel_size=(1, 1),
                            padding='same', 
                            activation=None,
                            name="rpn_L2_bbox_pred",
                            )
    self.rpn_L3_conv = tf.keras.layers.Conv2D(filters=512,
                            kernel_size=(3, 3),
                            activation=tf.nn.relu,
                            name="rpn_L3_conv",
                            padding="same",
                            )
    self.rpn_L3_cls_score = tf.keras.layers.Conv2D(filters=self.num_classes,
                            kernel_size=(1, 1),
                            activation=None,
                            name="rpn_L3_cls_score",
                            )
    self.rpn_L3_bbox_pred = tf.keras.layers.Conv2D(filters=self.num_classes*4,
                            kernel_size=(1, 1),
                            padding='same', 
                            activation=None,
                            name="rpn_L3_bbox_pred",
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
    l2_score = self.rpn_L2_cls_score(l2_feat)
    l2_bbox = self.rpn_L2_bbox_pred(l2_feat)
    l3_score = self.rpn_L3_cls_score(l3_feat)
    l3_bbox = self.rpn_L3_bbox_pred(l3_feat)
    y_pred = {
      "l1_score" : l1_score,
      "l1_bbox" : l1_bbox,
      "l2_score" : l2_score,
      "l2_bbox" : l2_bbox,
      "l3_score" : l3_score,
      "l3_bbox" : l3_bbox,
    }
    return y_pred

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
    labels = labelLayer(pred_score.shape[-1],pred_score.shape[1:3],y_true,self.imge_size)
    labels = tf.reshape(labels,[-1])
    select = tf.reshape(tf.where(tf.not_equal(labels, -1)),[-1])
    score = tf.gather(tf.reshape(pred_score,[-1]), select)
    labels = tf.gather(labels,select)
    return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score, labels=labels))
    
  def call(self, y_true, y_pred):
    if(self.gtformat=='xywh'):
      # convert [class, xstart, ystart, w, h] to
      # [y1, x1, y2, x2]
      gt_boxes = xywh2yxyx(y_true[:,1:])
      y_true = tf.stack([y_true[:,0],gt_boxes[:,0],gt_boxes[:,1],gt_boxes[:,2],gt_boxes[:,3]],axis=1)

    l1_loss = self._label_loss(y_pred["l1_score"],y_true)
    l2_loss = self._label_loss(y_pred["l2_score"],y_true)
    l3_loss = self._label_loss(y_pred["l3_score"],y_true)
    
    return l1_loss + l2_loss + l3_loss