import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tflib.img_tools import overlap_tf
from tflib.bbox_transform import xywh2yxyx

def draw_boxes(img,boxes,gtformat='xywh'):
  """
  Args:
    image: (image_num,H,W,1 or 3) tensor
    boxes: list of (box_num,4) tensor where 4 is
      [y_min, x_min, y_max, x_max]
    gtformat: xywh or yxyx
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