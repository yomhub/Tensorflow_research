# vgg16: Release
# resnet: wait
# roi layer: Alpha
# roi loss function: alpha
# rcnn loss function: current building
# proposal_target_layer_tf: current building
# global loss function: alpha
from __future__ import absolute_import
from __future__ import print_function

import os, sys
import tensorflow as tf
import numpy as np
import math
from layer_utils.generate_anchors import generate_anchors
# from layer_utils.proposal_layer import proposal_layer_tf
from layer_utils.snippets import generate_anchors_pre, generate_anchors_pre_tf
from layer_utils.proposal_top_layer import proposal_top_layer, proposal_top_layer_tf
# from layer_utils.anchor_target_layer import anchor_target_layer
from model.bbox_transform import bbox_transform_inv, clip_boxes, bbox_transform_inv_tf, clip_boxes_tf
from model.config import cfg
from tflib.anchor_target_layer import anchor_target_layer_tf
from tflib.proposal_target_layer import proposal_target_layer_tf
from tflib.common import *

_LOCAL_DIR = os.path.split(__file__)[0]
_DATASET_DIR= os.path.join(_LOCAL_DIR,'../','Dataset')
_NETS_DIR=os.path.join(_LOCAL_DIR,'../','Dataset','pre-trained-nets')
_NETS=['resnet_v2_152']
_IOU_POSITIVE_THRESHOLD = 0.7
_IOU_NEGATIVE_THRESHOLD = 0.3
_MAX_OUTPUTS_NUM = 400 

class Faster_RCNN(tf.keras.Model):
  """
  Args:
    num_classes: total number of class
    feature_layer_name: feature layer name
      one of'vgg16','resnet'
    anchor_size: anchor size factor
      default is 16  
    anchor_multiple: anchor multiple
      default is [1,3,5]
    anchor_ratio: anchor ratio, can be float and list
      default is [1,0.5,2]
    anchors will be:
      [anchor_ratio] * anchor_size * [anchor_multiple]
      and normalize as
      [-x/2,-y/2,x/2,y/2]
      only need to ADD center coordinate to anchor
      then we can get correspond window
    rpn_size: feature layer output
      default is [64,64]
    cls_in_size: [crop_height, crop_width]
      input size of classification

  Return: 
    Dictionary object: {
      ---RPN outouts---
      "rpn_bbox_pred": Tensor with shape (1,f_h,f_w,anchor_num*4)
        where 4 is [y1, x1, y2, x2]
      "rpn_cls_score": Tensor with shape (1,f_h,f_w,anchor_num*2)
        where 2 is [positive, negative]

      ---RCNN outouts---
      "bbox_pred": Tensor with shape (max_outputs_num, num_classes*4)
        where 4 is [y1, x1, y2, x2]
      "cls_score": Tensor with shape (max_outputs_num, num_classes)

      ---Normal output---
      "rois": Tensor with shape (max_outputs_num, 5)
        where 5 is [0, y1, x1, y2, x2]
      "rpn_scores": Tensor with shape (max_outputs_num)

      ---For Training---
      "img_sz" : image size
    }

  """

  def __init__(self,
    num_classes,
    feature_layer_name='vgg16',
    anchor_size=16,
    anchor_multiple=[1,3,5],
    anchor_ratio=[1,0.5,2],
    rpn_size=[64,64],
    max_outputs_num=400,
    nms_thresh=0.5,
    cls_in_size=[7,7],
    ):
    super(Faster_RCNN, self).__init__()
    # self.name='Faster_RCNN'
    self.max_outputs_num=int(max_outputs_num)
    self.nms_thresh=float(nms_thresh)
    if(feature_layer_name.lower()=='resnet'):
      self.feature_layer_name='resnet'
      self.feature_layer_chs = 512
    else:
      self.feature_layer_name='vgg16'
      self.feature_layer_chs = 512
      # final chs is 512
      # self.rpn_size=[,512]

    # [x1,y1,x2,y2]
    self.anchors = tf.constant(generate_anchors(
      base_size=anchor_size,
      ratios=np.array(anchor_ratio), 
      scales=np.array(anchor_multiple)), 
      dtype=tf.float32
      )
    self.base_size = anchor_size
    self.anchor_scales = all2list(anchor_multiple)
    self.anchor_ratio = all2list(anchor_ratio)
    self.rpn_size = all2list(rpn_size,2)
    self.cls_in_size = all2list(cls_in_size,2)
    self.num_classes = int(num_classes)
    # for rat in anchor_ratio:
    #   for sz in anchor_size:
    #     self.anchors.append([int(sz*rat),int(sz/rat)])

  def build(self, 
    input_shape,
    ):
    if(self.feature_layer_name=='vgg16'):
      vgg16=tf.keras.applications.VGG16(weights='imagenet', include_top=False)
      self.feature_model = tf.keras.models.Sequential([
        # tf.keras.Input((1024,1024,3)),
        # vgg16.get_layer("input_1"),
        # Original size
        vgg16.get_layer("block1_conv1"), vgg16.get_layer("block1_conv2"), vgg16.get_layer("block1_pool"),
        # Original size / 2
        vgg16.get_layer("block2_conv1"), vgg16.get_layer("block2_conv2"), vgg16.get_layer("block2_pool"),
        # Original size / 4
        vgg16.get_layer("block3_conv1"), vgg16.get_layer("block3_conv2"), vgg16.get_layer("block3_conv3"), 
        # Original size / 4
        vgg16.get_layer("block3_pool"),
        # Original size / 8
        vgg16.get_layer("block4_conv1"), vgg16.get_layer("block4_conv2"), vgg16.get_layer("block4_conv3"),
        # Original size / 8
        vgg16.get_layer("block4_pool"),
        # Original size / 16
        vgg16.get_layer("block5_conv1"), vgg16.get_layer("block5_conv2"), vgg16.get_layer("block5_conv3"),
        # Original size / 16
        # vgg16.get_layer("block5_pool"),
        # Original size / 32
      ],
      name=self.feature_layer_name
      )
    elif(self.feature_layer_name=='resnet'):
      rn=tf.keras.applications.resnet_v2.ResNet101V2()
      self.feature_model = tf.keras.models.Sequential([
        rn.get_layer("conv1_pad"), rn.get_layer("conv1_conv"), rn.get_layer("pool1_pad"), rn.get_layer("pool1_pool"),
        rn.get_layer("block2_conv1"), rn.get_layer("block2_conv2"), rn.get_layer("block2_pool"),
        rn.get_layer("block3_conv1"), rn.get_layer("block3_conv2"), rn.get_layer("block3_conv3"), 
        rn.get_layer("block3_pool"),
        rn.get_layer("block4_conv1"), rn.get_layer("block4_conv2"), rn.get_layer("block4_conv3"),
        rn.get_layer("block4_pool"),
        rn.get_layer("block5_conv1"), rn.get_layer("block5_conv2"), rn.get_layer("block5_conv3"),
        rn.get_layer("block5_pool"),
      ],
      name=self.feature_layer_name
      )
    # =====For RPN layer=====
    self.rpn_conv = tf.keras.layers.Conv2D(filters=self.feature_layer_chs,
                            kernel_size=(3, 3),
                            activation=tf.nn.relu,
                            name="rpn_conv/3x3",
                            )
    # 2 for have / not have obj
    self.rpn_cls_score = tf.keras.layers.Conv2D(filters=len(self.anchors)*2,
                            kernel_size=(1, 1),
                            activation=None,
                            name="rpn_cls_score",
                            )
    # 4 fro coordinate regression
    self.rpn_bbox_pred = tf.keras.layers.Conv2D(filters=len(self.anchors)*4,
                            kernel_size=(1, 1),
                            padding='VALID', 
                            activation=None,
                            name="rpn_bbox_pred",
                            )

    # For Classification layer
    # For each RPN output point, caculate num_classes class (eg: positive negative = 2 class)
    self.cls_layer = tf.keras.layers.Dense(self.num_classes,
                            input_shape=(1,self.cls_in_size[0]*self.cls_in_size[1]*self.feature_layer_chs),
                            activation=None,
                            name="classification"
                            )
    # For each RPN output point, caculate 4 regression value
    self.cls_bbox_layer = tf.keras.layers.Dense(self.num_classes*4,
                            input_shape=(1,self.cls_in_size[0]*self.cls_in_size[1]*self.feature_layer_chs),
                            activation=None,
                            name="classification_bbox"
                            )

    # super(Faster_RCNN, self).build(input_shape)

  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, im_info):
    """
    Coordinate order is [y1, x1, y2, x2] is
    diagonal pair of box corners.
    rpn_cls_prob: bbox score [1,y,x,2N], 
      where 2N is [negative, positive]
    rpn_bbox_pred: bbox coordinate [1, y, x, 4N]
      where 4N is [dx, dy, dw, dh]
      which are coefficients
    x1 = start_x + dx * (end_x - start_x + 1)
    y1 = start_y + dy * (end_y - start_y + 1)
    width = exp(dw) * (end_x - start_x + 1)
    height = exp(dh) * (end_y - start_y + 1)
    
    Input:
      im_info: img shape [y,x]

    Return:
      bboxs: shape (self.max_outputs_num,5) with [0, y1, x1, y2, x2] 
        coordinate in oringinal image and 0 is placeholder
      scores: shape(self.max_outputs_num,1)
    """
    feat_h = rpn_cls_prob.shape[1]
    feat_w = rpn_cls_prob.shape[2]
    stride_h=int(im_info[0]/feat_h)
    stride_w=int(im_info[1]/feat_w)
    shift_y = tf.range(feat_h) * stride_h
    shift_x = tf.range(feat_w) * stride_w 
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    sx = tf.reshape(shift_x, shape=(-1,))
    sy = tf.reshape(shift_y, shape=(-1,))
    shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))
    shifts = tf.transpose(tf.reshape(shifts, shape=[1, -1, 4]), perm=(1, 0, 2))
    shifts = tf.cast(shifts,tf.float32)

    # anchors: [x1,y1,x2,y2]
    anchors = tf.reshape(
      tf.add(
        tf.reshape(self.anchors,(1,-1,4)), 
        shifts),
      shape=(-1, 4)
      )
      
    rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))

    # proposals: [x1,y1,x2,y2]
    proposals = bbox_transform_inv_tf(anchors, rpn_bbox_pred)
    proposals = clip_boxes_tf(proposals, im_info)
    # change proposals to [y1,x1,y2,x2] 
    proposals = tf.stack(
      [proposals[:,1],
      proposals[:,0],
      proposals[:,3],
      proposals[:,2]
      ],
      axis=1
      )

    # only consider postive
    rpn_cls_prob = rpn_cls_prob[:,:,:,len(self.anchors):]
    rpn_cls_prob = tf.reshape(rpn_cls_prob, shape=(-1,))

    # put all 9 boxs per pixel
    indices,scores = tf.image.non_max_suppression_with_scores(
      boxes=proposals, 
      scores=rpn_cls_prob,
      max_output_size=self.max_outputs_num,
      iou_threshold=self.nms_thresh
      )
    bboxs = tf.gather(
      params=proposals,
      indices=indices,
      axis=0
      )

    # add class placeholder (0) before bboxs
    bboxs = tf.pad(bboxs,[[0,0],[1,0]])

    return bboxs,scores

  def _suit_fc_input(self,target,rpn_size):
    """
    SPN implementation.
    Convert target to rpn_size size.
    Arg:
      target: Tensor with (*, h, w, *)
      rpn_size: target [height, width]
    """
    # pooling shape heigher than rpn_size to FC input
    if(target.shape[2]>rpn_size[0] or
      target.shape[1]>rpn_size[1]):
      # resize RPN output to (self.rpn_size[0],self.rpn_size[1]) though max poolling
      kx_sz=int(math.ceil(target.shape[2]/rpn_size[0]))
      ky_sz=int(math.ceil(target.shape[1]/rpn_size[1]))
      target=tf.nn.max_pool(target,
                  ksize=[1,ky_sz,kx_sz,1],
                  strides=[1,ky_sz,kx_sz,1],
                  padding='SAME',
      )

    # padding shape lower than rpn_size to FC input 
    if(target.shape[2]<rpn_size[0] or
      target.shape[1]<rpn_size[1]):

      target=tf.image.pad_to_bounding_box(
        target,
        offset_height=int((rpn_size[1]-target.shape[1])/2) 
          if target.shape[1]<rpn_size[1] else 
          0,
        offset_width=int(
          (rpn_size[0]-target.shape[2])/2) 
          if target.shape[2]<rpn_size[0] else
          0,
        target_height=rpn_size[1],
        target_width=rpn_size[0]
      )

    return target

  def _crop_pool_layer(self, rois, in_size, rpn_feature):
    """
      Generate roi image upon features and pooling
      [y1,x1,y2,x2] coordinate in oringinal image
      mapping y1,x1,y2,x2 into (0,1)
      Args:
        rois: Tensor with (N, 4) where 4 is
          [y1,x1,y2,x2]
        rpn_feature: Tensor with (1, height, width, chs)
        in_size: input image [height, width]
    """
    rois_fet_size = tf.stack(
      [rois[:,0]/in_size[0],
      rois[:,1]/in_size[1],
      rois[:,2]/in_size[0],
      rois[:,3]/in_size[1]
      ],
      axis=1
      )

    roi_feat = tf.image.crop_and_resize(
      rpn_feature,
      boxes=rois_fet_size,
      # box_indices: each boxes ref index in rpn_feature.shape[0]
      box_indices=tf.zeros([rois_fet_size.shape[0]],dtype=tf.int32),
      crop_size=[self.cls_in_size[0]*2,self.cls_in_size[1]*2],
    )

    roi_feat = tf.nn.max_pool(roi_feat,
                  ksize=[1,2,2,1],
                  strides=[1,2,2,1],
                  padding='SAME',
    )

    return roi_feat

  def _region_proposal(self, feature, in_size):
    """
    Return:
      rois: Tensor with (self.max_outputs_num, 5)
        where 5 is [0, y1, x1, y2, x2] 
        coordinate in oringinal image and 0 is placeholder
      rpn_scores: Tensor with (self.max_outputs_num,1)
      ...
    """
    # padding inage with arounded zeros
    # so 3*3 conv will get same shape with feature layer
    rpn_feature = self.rpn_conv(
      tf.image.pad_to_bounding_box(
        feature,
        offset_height=1,
        offset_width=1,
        target_height=feature.shape[1]+2,
        target_width=feature.shape[2]+2,
      )
    )
    rpn_cls_score = self.rpn_cls_score(rpn_feature)
    # for every feature point, we get datas upon every chanels
    # index and possibilities be normalized upon global

    # shape = (1,y,x,2*anchor_num): [negative, positive]
    rpn_cls_prob = tf.nn.softmax(rpn_cls_score)
    # and box shape
    # shape = (1,y,x,4*anchor_num): [dy1, dx1, dy2, dx2]
    # where y1, x1, y2, x2 is diagonal pair of box corners
    rpn_bbox_pred = self.rpn_bbox_pred(rpn_feature)
    
    # rpn_cls_prob = self.cls_score_fc(rpn_cls_prob)
    # rpn_bbox_pred = self.bbox_regression(rpn_bbox_pred)

    # get highest score = (y,x)
    rpn_cls_pred = tf.reshape(
      tf.argmax(rpn_cls_prob,axis=3),
      (rpn_cls_prob.shape[1],
      rpn_cls_prob.shape[2])
    )

    rois, rpn_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, in_size)

    return rois, rpn_scores, rpn_cls_score, rpn_cls_prob, rpn_cls_pred, rpn_bbox_pred

  def _region_classification(self, roi_feat):
    roi_feat = tf.reshape(roi_feat,[roi_feat.shape[0],-1])
    cls_score = self.cls_layer(roi_feat)
    cls_pred = tf.nn.softmax(cls_score)
    # get highest score 
    cls_prob = tf.argmax(cls_score,axis=1)
    bbox_pred = self.cls_bbox_layer(roi_feat)
    return cls_score, cls_pred, cls_prob, bbox_pred

  def call(self, inputs):
    """
    Input: 
      image tensor with shape (batch,h,w,chs)
    Return: 
      Dictionary object: {
        ---RPN outouts---
        "rpn_bbox_pred": Tensor with shape (1,f_h,f_w,anchor_num*4)
          where 4 is [y1, x1, y2, x2]
        "rpn_cls_score": Tensor with shape (1,f_h,f_w,anchor_num*2)
          where 2 is [positive, negative]

        ---RCNN outouts---
        "bbox_pred": Tensor with shape (max_outputs_num, num_classes*4)
          where 4 is [y1, x1, y2, x2]
        "cls_score": Tensor with shape (max_outputs_num, num_classes)

        ---Normal output---
        "rois": Tensor with shape (max_outputs_num, 5)
          where 5 is [0, y1, x1, y2, x2]
        "rpn_scores": Tensor with shape (max_outputs_num)

        ---For Training---
        "img_sz" : image size
      }
    """
    if(inputs.dtype!=tf.float32 or inputs.dtype!=tf.float64):
      inputs = tf.cast(inputs,tf.float32)
    if(inputs.shape[0]==None):
      # import pdb; pdb.set_trace()
      return {}
    in_size = inputs.shape[1:3]
    feature = self.feature_model(inputs)
    rois, rpn_scores, rpn_cls_score, rpn_cls_prob, rpn_cls_pred, rpn_bbox_pred \
     = self._region_proposal(feature,in_size)
    roi_feat = self._crop_pool_layer(rois, in_size, feature)
    cls_score, cls_pred, cls_prob, bbox_pred = self._region_classification(roi_feat)

    y_pred = {
      "rpn_bbox_pred" : rpn_bbox_pred,
      "rpn_cls_score" : rpn_cls_score,
      "bbox_pred" : bbox_pred,
      "cls_score" : cls_score,
      "rois" : rois,
      "rpn_scores" : rpn_scores,
      "img_sz" : in_size,
      "num_classes" : self.num_classes,
    }
    return y_pred

class RCNNLoss(tf.keras.losses.Loss):
  """
  Args:
    sigma_rpn: sigma in RPN bbox loss
  Inputs: dictionary format
    y_true: {
      "img_sz": image size [height，width],
      "texts": Tensor with (N,5) and 5 is 
        [xstart, ystart, w, h, theta]
    }

    y_pred: {
      "rois": Tensor with shape (N,4)
        where 4 is [y1, x1, y2, x2]
      "rpn_scores": Tensor with shape (N,1)
    }
  """
  def __init__(self, cfg, cfg_name, sigma_rpn=3.0):    
    try:
      self.cfg=cfg[cfg_name]
    except:
      self.cfg=cfg['TRAIN']
    self.sigma_rpn=sigma_rpn
    self.loss_detail={}
    super(RCNNLoss, self).__init__()
  
  def _smooth_l1_loss(self, 
    bbox_pred, 
    bbox_targets, 
    bbox_inside_weights=1.0, 
    bbox_outside_weights=1.0, 
    sigma=1.0, 
    dim=[1]
  ):
    """
    Calculate sum(smooth_l1) 
      where smooth_l1=0.5(|x|^2) if |x|<1 else |x|-0.5
    Args:
      bbox_pred with shape (M,4)
      bbox_targets with shape (N,4)
      bbox_inside_weights with shape (N,4)
      bbox_outside_weights with shape (N,4)
    Return:
      loss value
    """
    assert bbox_pred.shape==bbox_targets.shape
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(tf.cast(tf.less(abs_in_box_diff, 1. / sigma_2),dtype=tf.float32))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(
      out_loss_box,
      axis=dim
    ))
    return loss_box

  def call(self, y_true, y_pred):
    """
    Return the sum of 
      cross_entropy + loss_box +
      rpn_cross_entropy + rpn_loss_box

    Inputs:
      y_true: Tensor with shape (total_gts,5)
        where 5 is [class, xstart, ystart, w, h]
        and class is class number
    
      y_pred: {
        ---RPN outouts---
        "rpn_bbox_pred": Tensor with shape (1,f_h,f_w,anchor_num*4)
          where 4 is [y1, x1, y2, x2]
        "rpn_cls_score": Tensor with shape (1,f_h,f_w,anchor_num*2)
          where 2 is [positive, negative]

        ---RCNN outouts---
        "bbox_pred": Tensor with shape (max_outputs_num, num_classes*4)
          where 4 is [y1, x1, y2, x2]
        "cls_score": Tensor with shape (max_outputs_num, num_classes)

        ---Normal output---
        "rois": Tensor with shape (max_outputs_num, 5)
          where 5 is [0, y1, x1, y2, x2]
        "rpn_scores": Tensor with shape (max_outputs_num)

        ---For Training---
        "img_sz" : image size
        "num_classes": int, total num of class
      }

    """

    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights \
     = anchor_target_layer_tf(
       all_anchors=y_pred["rpn_bbox_pred"], 
       gt_boxes=y_true[:,1:5], 
       im_info=y_pred["img_sz"], 
       settings=self.cfg,
       )

    # RPN, class loss
    rpn_select = tf.where(tf.not_equal(rpn_labels, -1))
    rpn_cls_score = tf.reshape(y_pred['rpn_cls_score'],[-1,2])
    rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select),[-1,2])
    rpn_label = tf.reshape(tf.gather(rpn_labels, rpn_select),[-1])

    rpn_cross_entropy = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

    # RPN, bbox loss
    rpn_bbox_pred = tf.reshape(y_pred['rpn_bbox_pred'],[-1,4])
    rpn_loss_box = self._smooth_l1_loss(
      bbox_pred=rpn_bbox_pred, 
      bbox_targets=rpn_bbox_targets, 
      bbox_inside_weights=rpn_bbox_inside_weights,
      bbox_outside_weights=rpn_bbox_outside_weights, 
      sigma=self.sigma_rpn,
    )

    rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights \
      = proposal_target_layer_tf(y_pred["rois"], y_pred["rpn_scores"], 
      y_true, y_pred["num_classes"], self.cfg)

    # RCNN, class loss
    cls_score = y_pred["cls_score"]
    label = tf.reshape(labels, [-1])
    cross_entropy = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=cls_score, 
        labels=label
      )
    )

    # RCNN, bbox loss
    bbox_pred = y_pred['bbox_pred']
    loss_box = self._smooth_l1_loss(
      bbox_pred=bbox_pred, 
      bbox_targets=bbox_targets, 
      bbox_inside_weights=bbox_inside_weights,
      bbox_outside_weights=bbox_outside_weights,
    )
    self.loss_detail={
      "cross_entropy":cross_entropy,
      "loss_box":loss_box,
      "rpn_cross_entropy":rpn_cross_entropy,
      "rpn_loss_box":rpn_loss_box,
    }
    return cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box