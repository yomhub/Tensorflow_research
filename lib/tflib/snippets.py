import numpy as np
import tensorflow as tf

@tf.function
def generate_real_anchors(anchors,img_size,layer_size):
  """
    Args:
      anchors: (N,4) with [x1,y1,x2,y2]
      img_size: original image size (h,w)
      layer_size: target layer size (h,w)
  """
  feat_h = layer_size[0]
  feat_w = layer_size[1]
  stride_h=int(img_size[0]/feat_h)
  stride_w=int(img_size[1]/feat_w)
  shift_y = tf.range(feat_h) * stride_h + int(stride_h/2)
  shift_x = tf.range(feat_w) * stride_w + int(stride_w/2)
  shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
  sx = tf.reshape(shift_x, shape=(-1,))
  sy = tf.reshape(shift_y, shape=(-1,))
  shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))
  shifts = tf.transpose(tf.reshape(shifts, shape=[1, -1, 4]), perm=(1, 0, 2))
  shifts = tf.cast(shifts,tf.float32)
  # anchors: [x1,y1,x2,y2]
  anchors = tf.reshape(
    tf.add(
      tf.reshape(anchors,(1,-1,4)), 
      shifts),
    shape=(-1, 4)
    )
  return anchors

@tf.function
def score_convert(scores):
  """
    Args: scores with (..,2*N)
      where 2N is [negative...,positive...]
    Return: scores with (-1,2)
      where 2 is [negative,positive]
  """
  N = int(scores.shape[-1]/2)
  scores = tf.reshape(scores,[-1,scores.shape[-1]])
  scores_neg = tf.reshape(scores[:,:N],[-1])
  scores_pos = tf.reshape(scores[:,N:],[-1])
  scores = tf.stack([scores_neg,scores_pos],axis=-1)
  return scores