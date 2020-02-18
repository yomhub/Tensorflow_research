import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tflib.img_tools import overlap_tf
from tflib.bbox_transform import xywh2yxyx

def draw_boxes(img,boxes,gtformat='xywh'):
  """
    Args:
      image: (image_num,H,W,1 or 3) tensor
      boxes: list of (box_num,4) tensor where 4 is
        xywh with [xstart, ystart, width, height]
        yxyx with [y_min, x_min, y_max, x_max]
      gtformat: xywh or yxyx
    Return:
      drawed image tensor with (image_num,H,W,1 or 3) 
  """
  imgh=float(img.shape[-3])
  imgw=float(img.shape[-2])
  if(len(img.shape)==3):
    img = tf.reshape(img,[1,]+img.shape)
  if(type(boxes)!=list):
    boxes = [boxes]
  if(img.dtype!=tf.float32 or img.dtype!=tf.float64):
    # tf.image.draw_bounding_boxes need float32 image
    img = tf.cast(img,tf.float32)
  ret = []
  for i in range(len(boxes)):
    if(gtformat=='xywh'):
      box = tf.reshape(xywh2yxyx(boxes[i][:,-4:]),[1,-1,4])
    else:
      box = tf.reshape(boxes[i],[1,-1,4])
    if(tf.reduce_max(box)>1.0):
      box = tf.stack([box[0,:,0]/imgh,box[0,:,1]/imgw,box[0,:,2]/imgh,box[0,:,3]/imgw],axis=1)
      box = tf.reshape(box,[1,-1,4])
    col = tf.random.uniform(
      (box.shape[-2],3),
      minval=0,
      maxval=254,
      )
    ret += [tf.image.draw_bounding_boxes(tf.reshape(img[i],[1,]+img[i].shape[-3:]),box,col)]
  ret = tf.convert_to_tensor(ret)

  return tf.reshape(ret,[ret.shape[0],]+ret.shape[-3:])

def draw_grid_in_gt(layer_shape,gt_box,image):
  """
    Draw grid inside GT box refer to layer's receptive field
    Args:
      layer_shape: [h,w]
      gt_box: (N,4 or 5) with [(label),y1,x1,y2,x2]
      image: tensor with shape (h,w,ch) or (1,h,w,ch)
    Return:
      image tensor with shape (1,h,w,ch)
  """
  if(len(image.shape)==3):
    image = tf.reshape(image,[1,]+image.shape)
  imgh,imgw = float(image.shape[-3]),float(image.shape[-2])
  cube_h,cube_w = image.shape[-3]/layer_shape[0],image.shape[-2]/layer_shape[1]
  boxes = gt_box[:,-4:]
  ret = image
  for box in boxes:
    x_start,x_end = math.ceil(box[1] / cube_w),math.floor(box[3] / cube_w)
    y_start,y_end = math.ceil(box[0] / cube_h),math.floor(box[2] / cube_h)
    sub_box_x = tf.range(x_start,x_end+1,dtype=tf.float32)*cube_w
    sub_box_y = tf.range(y_start,y_end+1,dtype=tf.float32)*cube_h
    sub_box_xs = tf.concat([tf.reshape(box[1],[1]),sub_box_x],axis=0)
    sub_box_xe = tf.concat([sub_box_x,tf.reshape(box[3],[1])],axis=0)
    sub_box_ys = tf.concat([tf.reshape(box[0],[1]),sub_box_y],axis=0)
    sub_box_ye = tf.concat([sub_box_y,tf.reshape(box[2],[1])],axis=0)
    sub_box_xs,sub_box_ys = tf.meshgrid(sub_box_xs,sub_box_ys)
    sub_box_xe,sub_box_ye = tf.meshgrid(sub_box_xe,sub_box_ye)
    sub_box = tf.stack([sub_box_ys/imgh,sub_box_xs/imgw,sub_box_ye/imgh,sub_box_xe/imgw],axis=-1)
    ret = tf.image.draw_bounding_boxes(ret,tf.reshape(sub_box,[1,-1,4]),tf.convert_to_tensor([[36.0,128.0,128.0]]))
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

def load_and_preprocess_image(imgdir, outsize=None):
  fd=""
  if isinstance(imgdir,str):
    fd=imgdir
  else:
    fd=str(imgdir.numpy())
  image = tf.image.decode_image(tf.io.read_file(fd))
  if(outsize!=None):
    image = tf.image.resize(image, outsize)
  return image

def check_nan(tar):
  if(type(tar)==list):
    inc_list = []
    for i in range(len(tar)):
      if(tar[i]==None):
        continue
      inc=tf.where(tf.math.is_nan(tar[i]))
      if(inc.shape[0]!=0):
        inc_list.append([i,inc.shape[0]])
    if(len(inc_list)!=0):
      return inc_list
  else:
    inc=tf.where(tf.math.is_nan(tar))
    if(inc.shape[0]!=0):
      return inc.shape[0]
  return 0