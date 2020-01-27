import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tflib.overlap import overlap_tf

def draw_boxes(img,box):
  """
  Args:
    image: (image_num,H,W,1 or 3) tensor
    box: (image_num,box_num,4) tensor where 4 is
      [y_min, x_min, y_max, x_max]
  """
  if(len(img.shape)==3):
    img = tf.reshape(img,[1,]+img.shape)
  if(len(box.shape)==2):
    box = tf.reshape(box,[1,]+box.shape)
  if(img.dtype!=tf.float32 or img.dtype!=tf.float64):
    img = tf.cast(img,tf.float32)
  maxs = tf.reduce_max(box,axis=2)
  maxs = tf.reduce_max(maxs,axis=1)
  if(maxs>1.0):
    h=img.shape[1]
    w=img.shape[2]
    box=tf.stack(
      [box[:,:,0]/h,box[:,:,1]/w,box[:,:,2]/h,box[:,:,3]/w],
      axis=2
      )
  col = tf.random.uniform(
    (box.shape[1],3),
    minval=0,
    maxval=254,
    )

  ret = tf.image.draw_bounding_boxes(img,box,col)
  return ret

def show_img(img):
  if(len(img.shape)>3):
    img = tf.reshape(img[0,:,:,:],img.shape[1:])
  if(img.dtype!=tf.uint8):
    img = tf.cast(img,tf.uint8)
  plt.imshow(img)
  plt.show()

def fn_rate(pred_box,gt_box):
  """
  Args:
    pred_box: (pred_num,4) tensor where 4 is
      [y1,x1,y2,x2]
    gt_box: (gt_box,4) tensor where 4 is
      [y1,x1,y2,x2]
  """
  overlap = overlap_tf(pred_box,gt_box)
  pred_overlap = tf.reduce_max(overlap,axis=1)
  gt_overlap = tf.reduce_max(overlap,axis=0)

  