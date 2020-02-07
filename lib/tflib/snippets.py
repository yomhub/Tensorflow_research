import numpy as np
import tensorflow as tf
from tflib.generate_anchors import generate_anchors

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

# @tf.function
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

def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8,16,32), anchor_ratios=(0.5,1,2)):
  """ A wrapper function to generate anchors given different scales
    Also return the number of anchors in variable 'length'
  """
  anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
  A = anchors.shape[0]
  shift_x = np.arange(0, width) * feat_stride
  shift_y = np.arange(0, height) * feat_stride
  shift_x, shift_y = np.meshgrid(shift_x, shift_y)
  shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
  K = shifts.shape[0]
  # width changes faster, so here it is H, W, C
  anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
  anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
  length = np.int32(anchors.shape[0])

  return anchors, length

def generate_anchors_pre_tf(height, width, feat_stride=[16,16], base_size=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
  shift_x = tf.range(width) * feat_stride[1] # width
  shift_y = tf.range(height) * feat_stride[0] # height
  shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
  sx = tf.reshape(shift_x, shape=(-1,))
  sy = tf.reshape(shift_y, shape=(-1,))
  shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))
  K = tf.multiply(width, height)
  shifts = tf.transpose(tf.reshape(shifts, shape=[1, K, 4]), perm=(1, 0, 2))

  anchors = generate_anchors(
    base_size=base_size,
    ratios=np.array(anchor_ratios), 
    scales=np.array(anchor_scales))
  A = anchors.shape[0]
  anchor_constant = tf.constant(anchors.reshape((1, A, 4)), dtype=tf.int32)

  length = K * A
  anchors_tf = tf.reshape(tf.add(anchor_constant, shifts), shape=(length, 4))
  
  return tf.cast(anchors_tf, dtype=tf.float32), length
