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
from layer_utils.snippets import generate_anchors_pre, generate_anchors_pre_tf
from model.config import cfg
from tflib.bbox_transform import bbox_transform_inv_tf, clip_boxes_tf, xywh2yxyx
from tflib.anchor_target_layer import anchor_target_layer_tf
from tflib.proposal_target_layer import proposal_target_layer_tf
from tflib.snippets import generate_real_anchors, score_convert
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
      outsize: feature layer output
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
    anchor_size=1,
    anchor_ratio=[1,2,0.5],
    bx_choose="nms",
    max_outputs_num=2000,
    nms_thresh=0.1,
    cls_in_size=[7,7],
    fc_node=1024,
    ):
    super(Faster_RCNN, self).__init__()
    # self.name='Faster_RCNN'
    self.max_outputs_num=int(max_outputs_num/2)
    self.nms_thresh=float(nms_thresh)
    if(feature_layer_name.lower()=='resnet'):
      self.feature_layer_name='resnet'
      self.feature_layer_chs = 512
    else:
      self.feature_layer_name='vgg16'
      self.feature_layer_chs = 512
      # final chs is 512

    self.base_size = anchor_size
    self.anchor_ratio = all2list(anchor_ratio)
    self.cls_in_size = all2list(cls_in_size,2)
    self.num_classes = int(num_classes)
    # for rat in anchor_ratio:
    #   for sz in anchor_size:
    #     self.anchors.append([int(sz*rat),int(sz/rat)])
    self.bx_choose = "top_k" if bx_choose == "top_k" else "nms"
    self.fc_node = int(fc_node)

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
          vgg16.get_layer('block3_pool').output,
          # vgg16.get_layer('block4_conv3').output,
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
    
    anchors = tf.constant(generate_anchors(
      base_size=8, # L3 Receptive field 8*8
      ratios=np.array(self.anchor_ratio), 
      scales=np.array([0.5,1,2])), 
      dtype=tf.float32
      )                
    self.rpn1_anchors = tf.constant(generate_real_anchors(anchors,[height,width],[int(height/8),int(width/8)]))
    self.rpn1_conv = tf.keras.models.Sequential(
      [
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=None, padding="same"),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.relu, padding="same"),
      ],
      name="rpn1_conv",
    )
    self.rpn1_cls_score = tf.keras.layers.Conv2D(filters=len(anchors)*2,
                            kernel_size=(1, 1),
                            activation=None,
                            name="rpn1_cls_score",
                            )

    self.rpn1_bbox_pred = tf.keras.layers.Conv2D(filters=len(anchors)*4,
                            kernel_size=(1, 1),
                            padding='VALID', 
                            activation=None,
                            name="rpn1_bbox_pred",
                            )

    # =====For RPN layer=====
    anchors = tf.constant(generate_anchors(
      base_size=16, # L4 Receptive field 16*16
      ratios=np.array(self.anchor_ratio), 
      scales=np.array([0.5,0.8,1.1])), 
      dtype=tf.float32
      )                
    self.anchors = tf.constant(generate_real_anchors(anchors,[height,width],[int(height/16),int(width/16)]))
    self.rpn_conv = tf.keras.layers.Conv2D(filters=self.feature_layer_chs,
                            kernel_size=(3, 3),
                            activation=tf.nn.relu,
                            name="rpn_conv/3x3",
                            padding="same",
                            )
    # 2 for have / not have obj
    self.rpn_cls_score = tf.keras.layers.Conv2D(filters=len(anchors)*2,
                            kernel_size=(1, 1),
                            activation=None,
                            name="rpn_cls_score",
                            )
    # 4 fro coordinate regression
    self.rpn_bbox_pred = tf.keras.layers.Conv2D(filters=len(anchors)*4,
                            kernel_size=(1, 1),
                            padding='VALID', 
                            activation=None,
                            name="rpn_bbox_pred",
                            )
                       
    # For Classification layer
    # fc6 + fc7
    self.fc6_layer = tf.keras.layers.Dense(self.fc_node,
                            input_shape=(1,self.cls_in_size[0]*self.cls_in_size[1]*self.feature_layer_chs),
                            activation=None,
                            name="fc6",
                            )
    self.fc7_layer = tf.keras.layers.Dense(self.fc_node,
                            input_shape=(1,self.cls_in_size[0]*self.cls_in_size[1]*self.feature_layer_chs),
                            activation=None,
                            name="fc7",
                            )
    self.fc6_dp_layer = tf.keras.layers.Dropout(0.5)
    self.fc7_dp_layer = tf.keras.layers.Dropout(0.5)
                            
    # For each RPN output point, caculate num_classes class (eg: positive negative = 2 class)
    self.cls_layer = tf.keras.layers.Dense(self.num_classes,
                            input_shape=(1,self.fc_node),
                            activation=None,
                            name="classification"
                            )
    # For each RPN output point, caculate 4 regression value
    self.cls_bbox_layer = tf.keras.layers.Dense(self.num_classes*4,
                            input_shape=(1,self.fc_node),
                            activation=None,
                            # activation="sigmoid",
                            name="classification_bbox"
                            )

    # super(Faster_RCNN, self).build(input_shape)

  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, anchors, im_info):
    """
      Coordinate order is [y1, x1, y2, x2] is
      diagonal pair of box corners.
      rpn_cls_prob: bbox score [1,y,x,2N], 
        where 2N is [negative..., positive...]
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
    rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))

    # proposals: [x1,y1,x2,y2]
    # return proposals to [y1,x1,y2,x2] 
    proposals = bbox_transform_inv_tf(anchors, rpn_bbox_pred)
    proposals = clip_boxes_tf(proposals, im_info)

    # only consider postive
    rpn_cls_prob = rpn_cls_prob[:,:,:,int(rpn_cls_prob.shape[-1]/2):]
    rpn_cls_prob = tf.reshape(rpn_cls_prob, shape=(-1,))

    # put all 9 boxs per pixel
    if(self.bx_choose=="top_k"):
      scores, indices = tf.nn.top_k(tf.reshape(rpn_cls_prob,[-1]), k=self.max_outputs_num)
      scores = tf.reshape(scores, shape=(-1, 1))
      bboxs = tf.gather(proposals, indices)
    else:
      indices = tf.image.non_max_suppression(
        boxes=proposals, 
        scores=rpn_cls_prob,
        max_output_size=self.max_outputs_num,
        iou_threshold=self.nms_thresh
        )
      # indices, _ = tf.unique(indices)
      scores = tf.gather(rpn_cls_prob,indices)
      bboxs = tf.gather(proposals, indices)
    
      if(indices.shape[0]!=None and self.max_outputs_num>indices.shape[0]):
        scores = tf.reshape(scores,[-1,1])
        scores = tf.pad(scores,[[0,self.max_outputs_num-indices.shape[0]],[0,0]],constant_values=-1)
        scores = tf.reshape(scores,[-1])
        bboxs = tf.pad(bboxs,[[0,self.max_outputs_num-indices.shape[0]],[0,0]],constant_values=-1)

    # add class placeholder (0) before bboxs
    bboxs = tf.pad(bboxs,[[0,0],[1,0]])

    return bboxs,scores

  def _suit_fc_input(self,target,outsize):
    """
      SPN implementation.
      Convert target to outsize size.
      Arg:
        target: Tensor with (*, h, w, *)
        outsize: target [height, width]
    """
    # pooling shape heigher than outsize to FC input
    if(target.shape[2]>outsize[1] or
      target.shape[1]>outsize[0]):
      # resize output to outsizethough max poolling
      kx_sz=int(math.ceil(target.shape[2]/outsize[1]))
      ky_sz=int(math.ceil(target.shape[1]/outsize[0]))
      target=tf.nn.max_pool(target,
                  ksize=[1,ky_sz,kx_sz,1],
                  strides=[1,ky_sz,kx_sz,1],
                  padding='SAME',
      )

    # padding shape lower than outsize to FC input 
    if(target.shape[2]<outsize[1] or
      target.shape[1]<outsize[0]):

      target=tf.image.pad_to_bounding_box(
        target,
        offset_height=int((outsize[0]-target.shape[1])/2) 
          if target.shape[1]<outsize[0] else 
          0,
        offset_width=int(
          (outsize[1]-target.shape[2])/2) 
          if target.shape[2]<outsize[1] else
          0,
        target_height=outsize[0],
        target_width=outsize[1]
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
      box_indices=tf.zeros((rois_fet_size.shape[0]),dtype=tf.int32),
      crop_size=[self.cls_in_size[0]*2,self.cls_in_size[1]*2],
      # crop_size=[self.cls_in_size[0],self.cls_in_size[1]],
    )
    
    roi_feat = tf.nn.max_pool(roi_feat,
                  ksize=[1,2,2,1],
                  strides=[1,2,2,1],
                  padding='SAME',
    )

    return roi_feat

  def _region_proposal(self, feature, l4feature, in_size):
    """
      Return:
        rois: Tensor with (self.max_outputs_num, 5)
          where 5 is [0, y1, x1, y2, x2] 
          coordinate in oringinal image and 0 is placeholder
        rpn_scores: Tensor with (self.max_outputs_num,1)
          where 1 is [positive]
        ...
    """
    rpn1_feature = self.rpn1_conv(l4feature)
    rpn1_cls_score = self.rpn1_cls_score(rpn1_feature)
    rpn1_bbox_pred = self.rpn1_bbox_pred(rpn1_feature)
    # rpn_cls_prob = [negative, negative, ..., positive, positive,...]
    rpn1_cls_prob = tf.nn.softmax(rpn1_cls_score,axis=3)
    rois1, rpn_scores1 = self._proposal_layer(rpn1_cls_prob, rpn1_bbox_pred, self.rpn1_anchors, in_size)

    # padding inage with arounded zeros
    # so 3*3 conv will get same shape with feature layer
    rpn_feature = self.rpn_conv(feature)
    rpn_cls_score = self.rpn_cls_score(rpn_feature)
    # for every feature point, we get datas upon every chanels
    # index and possibilities be normalized upon global

    # rpn_cls_prob = [negative, negative, ..., positive, positive,...]
    rpn_cls_prob = tf.nn.softmax(rpn_cls_score,axis=3)
    # and box shape
    # shape = (1,y,x,4*anchor_num): [dy1, dx1, dy2, dx2]
    # where y1, x1, y2, x2 is diagonal pair of box corners
    rpn_bbox_pred = self.rpn_bbox_pred(rpn_feature)
    
    # rpn_cls_prob = self.cls_score_fc(rpn_cls_prob)
    # rpn_bbox_pred = self.bbox_regression(rpn_bbox_pred)

    rois, rpn_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, self.anchors, in_size)

    rois = tf.concat([rois,rois1],axis=0)
    # (self.max_outputs_num,1), [positive]
    rpn_scores = tf.concat([rpn_scores,rpn_scores1],axis=0)

    return rois, rpn_scores, rpn_cls_score, rpn_bbox_pred, rpn1_cls_score, rpn1_bbox_pred

  def _region_classification(self, roi_feat, in_size):
    roi_feat = tf.reshape(roi_feat,
      [roi_feat.shape[0],roi_feat.shape[1]*roi_feat.shape[2]*roi_feat.shape[3]])
    # roi_feat = self.fc6_dp_layer(self.fc6_layer(roi_feat))
    roi_feat = self.fc6_layer(roi_feat)
    # roi_feat = self.fc7_dp_layer(self.fc7_layer(roi_feat))
    roi_feat = self.fc7_layer(roi_feat)
    cls_score = tf.clip_by_value(self.cls_layer(roi_feat),0.000001,1.0)
    cls_pred = tf.nn.softmax(cls_score)
    # get highest score 
    cls_prob = tf.argmax(cls_score,axis=1)
    bbox_pred = self.cls_bbox_layer(roi_feat)
    # bbox_pred = tf.multiply(bbox_pred,tf.convert_to_tensor(in_size[0],dtype=bbox_pred.dtype))
    return cls_score, cls_pred, cls_prob, bbox_pred

  # @tf.function
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

    front_feature,feature = self.feature_model(inputs)

    rois, rpn_scores, rpn_cls_score, rpn_bbox_pred, rpn_cls_score1, rpn_bbox_pred1 \
     = self._region_proposal(feature,front_feature,in_size)
    roi_feat = self._crop_pool_layer(rois, in_size, feature)
    cls_score, cls_pred, cls_prob, bbox_pred = self._region_classification(roi_feat,in_size)

    y_pred = {
      # RPN 
      # (1,f_h,f_w,anchor_num*4) [y1, x1, y2, x2]
      "rpn_bbox_pred" : rpn_bbox_pred,
      # (1,h,w,ancnum*2), [negative...,positive...]
      "rpn_cls_score" : rpn_cls_score,
      "rpn_cls_score1" : rpn_cls_score1,
      "rpn_bbox_pred1" : rpn_bbox_pred1,
      # RCNN output
      "bbox_pred" : bbox_pred,
      "cls_score" : cls_score,
      # after top k / nms
      "rois" : rois,
      # (self.max_outputs_num,1), [positive]
      "rpn_scores" : rpn_scores,

      "img_sz" : inputs.shape[1:3],
      "num_classes" : self.num_classes,
    }
    return y_pred

class RCNNLoss(tf.keras.losses.Loss):
  """
    Return the sum of 
      cross_entropy + loss_box +
      rpn_cross_entropy + rpn_loss_box
    Args:
      sigma_rpn: sigma in RPN bbox loss
    Inputs:
      y_true: Tensor with shape (total_gts,5)
        where 5 is [class, xstart, ystart, w, h] if gtformat=='xywh'
        or [class, y1, x1, y2, x2] if gtformat=='yxyx' or 'yx'
        and class is class number
    
      y_pred: {
        ---RPN outouts---
        "rpn_bbox_pred": Tensor with shape (1,f_h,f_w,anchor_num*4)
          where 4 is [y1, x1, y2, x2]
        "rpn_cls_score": Tensor with shape (1,f_h,f_w,anchor_num*2)
          where anchor_num*2 is [negative..., positive...]

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
  def __init__(self, cfg, cfg_name, sigma_rpn=3.0, gtformat='yxyx'):    
    try:
      self.cfg=cfg[cfg_name]
    except:
      self.cfg=cfg['TRAIN']
    self.sigma_rpn=sigma_rpn
    self.loss_detail={}
    if(gtformat=='yxyx' or gtformat=='yx'):
      self.gtformat='yxyx'
    else:
      self.gtformat='xywh'
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
    in_loss_box = tf.where(tf.less(abs_in_box_diff, 1. / sigma_2),
                            tf.pow(in_box_diff, 2) * (sigma_2 / 2.),
                            (abs_in_box_diff - (0.5 / sigma_2))
                            )
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(
      out_loss_box,
      axis=dim
    ))
    return loss_box

  def _rpn_loss(self,gt_boxes,rpn_bbox_pred,rpn_cls_score,img_sz):
    # convert [negative,...,positive,..] to [negitive,positive,...]
    rpn_cls_score = score_convert(rpn_cls_score)
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights \
     = anchor_target_layer_tf(
       all_anchors = rpn_bbox_pred, 
       gt_boxes = gt_boxes, 
       im_info = img_sz, 
       settings = self.cfg,
       )

    # RPN, class loss
    rpn_select = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)),[-1])
    rpn_cls_score = tf.gather(rpn_cls_score, rpn_select)
    rpn_label = tf.reshape(tf.gather(rpn_labels, rpn_select),[-1])

    rpn_cross_entropy = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

    # RPN, bbox loss
    rpn_bbox_pred = tf.reshape(rpn_bbox_pred,[-1,4])
    rpn_loss_box = self._smooth_l1_loss(
      bbox_pred=rpn_bbox_pred, 
      bbox_targets=rpn_bbox_targets, 
      bbox_inside_weights=rpn_bbox_inside_weights,
      bbox_outside_weights=rpn_bbox_outside_weights, 
      sigma=self.sigma_rpn,
    )

    return rpn_cross_entropy, rpn_loss_box

  def call(self, y_true, y_pred):
    if(self.gtformat=='xywh'):
      # convert [class, xstart, ystart, w, h] to
      # [y1, x1, y2, x2]
      gt_boxes = xywh2yxyx(y_true[:,1:])
      y_true = tf.stack([y_true[:,0],gt_boxes[:,0],gt_boxes[:,1],gt_boxes[:,2],gt_boxes[:,3]],axis=1)
    else:
      gt_boxes = y_true[:,1:]

    rpn_cross_entropy, rpn_loss_box = self._rpn_loss(
      gt_boxes=gt_boxes,
      rpn_bbox_pred=y_pred["rpn_bbox_pred"],
      rpn_cls_score=y_pred['rpn_cls_score'],
      img_sz=y_pred["img_sz"],
      )

    rpn_cross_entropy1, rpn_loss_box1 = self._rpn_loss(
      gt_boxes=gt_boxes,
      rpn_bbox_pred=y_pred["rpn_bbox_pred1"],
      rpn_cls_score=y_pred['rpn_cls_score1'],
      img_sz=y_pred["img_sz"],
      )
    
    bbox_pred, cls_score, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights \
      = proposal_target_layer_tf(
        rpn_rois=y_pred["rois"], rpn_scores=y_pred["rpn_scores"], 
        rcnn_rois=y_pred['bbox_pred'], rcnn_scores=y_pred["cls_score"],
        gt_boxes=y_true, num_classes=y_pred["num_classes"], settings=self.cfg)

    # RCNN, class loss
    label = tf.reshape(labels, [-1])
    cross_entropy = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=cls_score, 
        labels=label
      )
    )

    # RCNN, bbox loss
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
      "rpn_cross_entropy1":rpn_cross_entropy1,
      "rpn_loss_box1":rpn_loss_box1,
    }
    
    return cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box + rpn_cross_entropy1 + rpn_loss_box1