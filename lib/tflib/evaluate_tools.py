import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
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

def draw_msk_in_gt(msk,image):
  """
    Draw mask in gt box.
    Return: image
  """
  msk = tf.reshape(msk,[-1])
  col = tf.random.uniform([image.shape[-1]],minval=128,maxval=256,dtype=image.dtype)
  img_shape = image.shape
  image = image.numpy().reshape([-1,img_shape[-1]])
  image[msk>0]=col
  return tf.convert_to_tensor(image.reshape(img_shape))
  
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

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt 

@tf.function
def overlap_tf(xbboxs,ybboxs):
  """
    Inputs:
      xbboxs: tensor with shape (M,4)
      ybboxs: tensor with shape (N,4)
        where 4 is (y1,x1,y2,x2)
    Assert:
      xbboxs.shape[-1]==ybboxs.shape[-1]==4
    Output:
      scores: shape (M,N)
  """
  assert(xbboxs.shape[-1]==4 and ybboxs.shape[-1]==4)
  xbboxs_t = tf.transpose(
    tf.reshape(xbboxs,[1,xbboxs.shape[0],xbboxs.shape[1]]),
    [1,0,2])
  ybboxs_t = tf.reshape(ybboxs,[1,ybboxs.shape[0],ybboxs.shape[1]])
  # shape (M,N)
  lap_y = tf.minimum(xbboxs_t[:,:,2],ybboxs_t[:,:,2])-tf.maximum(xbboxs_t[:,:,0],ybboxs_t[:,:,0])+1.0
  lap_x = tf.minimum(xbboxs_t[:,:,3],ybboxs_t[:,:,3])-tf.maximum(xbboxs_t[:,:,1],ybboxs_t[:,:,1])+1.0
  area=lap_x*lap_y
  unlap = tf.where(
    tf.logical_and(lap_x>0,lap_y>0),
    tf.add(
      # xbox area
      (xbboxs_t[:,:,2]-xbboxs_t[:,:,0]+1.0)*(xbboxs_t[:,:,3]-xbboxs_t[:,:,1]+1.0),
      # ybox area
      (ybboxs_t[:,:,2]-ybboxs_t[:,:,0]+1.0)*(ybboxs_t[:,:,3]-ybboxs_t[:,:,1]+1.0)
    )-area
    ,0
    )
  score = tf.where(unlap>0,area/unlap,0)
  return score

def multi_overlap_pixel(xbboxs,ybboxs):
  """
    Inputs:
      xbboxs: LIST of tensor with shape (M,4)
        a area consist of M rectangles
      ybboxs: tensor with shape (N,4)
        where 4 is (y1,x1,y2,x2)
    Assert:
      xbboxs.shape[-1]==ybboxs.shape[-1]==4
    Output:
      LIST of scores tensor with shape (M,N)
  """
  if(type(xbboxs)!=list):
    xbboxs = [xbboxs]
  scores = []
  for xbx in xbboxs:
    overlap_pre = np.zeros((ybboxs.shape[0]))
    x_y1,x_x1 = tf.reduce_min(xbx[:,0:2]).numpy()
    x_y2,x_x2 = tf.reduce_max(xbx[:,2:4]).numpy()
    xbx = xbx.numpy()
    # (N)
    lap_y = tf.minimum(x_y2,ybboxs[:,2])-tf.maximum(x_y1,ybboxs[:,0])+1.0
    lap_x = tf.minimum(x_x2,ybboxs[:,3])-tf.maximum(x_x1,ybboxs[:,1])+1.0
    incs = tf.where(tf.logical_and(lap_x>0,lap_y>0)).numpy()
    if(incs.shape[0]==None):
      scores.append(tf.convert_to_tensor(overlap_pre))
      continue
    for inc in incs:
      ybx = ybboxs[inc].numpy()
      y1 = math.floor(min(ybx[0],x_y1))
      x1 = math.floor(min(ybx[1],x_x1))
      y2 = math.ceil(max(ybx[2],x_y2))
      x2 = math.ceil(max(ybx[3],x_x2))
      label = np.zeros((y2-y1,x2-x1),dtype=np.int8)
      for atom in xbx:
        label[int(atom[0]-y1):int(atom[2]-y1),int(atom[1]-x1):int(atom[3]-x1)]=1
      label[int(ybx[0]-y1):int(ybx[2]-y1),int(ybx[1]-x1):int(ybx[3]-x1)]+=1
      overlap_pre[inc] = np.sum(label>1)/np.sum(label>0)
    scores.append(tf.convert_to_tensor(overlap_pre))
    
  return scores


# @tf.function
def check_inside(boxes,img_size):
  """
    Args:
      boxes: (N,4) with [y1,x1,y2,x2]
      img_size: [height,width]
    Return: BOOL mask with (N) shape
  """
  cond = tf.logical_and(
    tf.logical_and(boxes[:,0]>=0.0 , boxes[:,0]<=img_size[0]),
    tf.logical_and(boxes[:,1]>=0.0 , boxes[:,1]<=img_size[1])
  )
  cond2 = tf.logical_and(
    tf.logical_and(boxes[:,2]>=0.0 , boxes[:,0]<=img_size[0]),
    tf.logical_and(boxes[:,3]>=0.0 , boxes[:,1]<=img_size[1])
  )
  return tf.logical_and(cond,cond2)

def random_gt_generate(img, gtbox, increment=10, multiple=None, max_box_pre=None):
  """
    Args:
      img: tensor with (N,h,w,c)
      gtbox: 
        list (N>=1) of tensor or single (N=1) tensor
        tensor shape (gt_box_num,5) where 5 is
        [label,y1,x1,y2,x2]
      increment: int, increase number pre img
      multiple: if multiple is given, function will multiply
        gt_box_num as increment pre img
      max_box_pre: int, max gtbox num pre img.
    Return:
      img (N,h,w,c), list of tensor
  """
  if(type(gtbox)!=list):
    gtbox=[gtbox]
  gtimg = img.numpy()
  imgh = gtimg.shape[1]
  imgw = gtimg.shape[2]
  box_list = []
  for i in range(len(gtbox)):
    boxes = gtbox[i].numpy()
    gt_num = boxes.shape[0]

    add_num = increment if (multiple==None) else int(gt_num*multiple)
    if(max_box_pre!=None):
      add_num = add_num if (add_num+gt_num) < max_box_pre else max_box_pre-gt_num
    if(add_num<=0):
      continue
    
    # onlu use box small then half original image
    low_half = np.where(boxes[:,2]-boxes[:,0]<imgh & boxes[:,3]-boxes[:,1]<imgw)[0].reshape([-1])
    if(low_half.shape[0]==0):
      continue
    low_half = np.take(boxes,low_half)
    txt_img = []
    for singletext in low_half:
      txt_img.append(gtimg[i,int(singletext[1]):int(singletext[3]),int(singletext[2]):int(singletext[4]),:])

    incs = np.random.randint(0,low_half.shape[0],(add_num))
    gen_box = np.take(low_half,incs)
    

    # [-0.5,0.5]
    dx = (np.random.random_sample((add_num))-0.5)*imgw
    dy = (np.random.random_sample((add_num))-0.5)*imgh

    gen_box[:,1:4:2] = np.where(gen_box[:,1]+dy<0 | gen_box[:,3]+dy>imgh,gen_box[:,1:4:2]-dy,gen_box[:,1:4:2]+dy)
    gen_box[:,2:5:2] = np.where(gen_box[:,2]+dy<0 | gen_box[:,4]+dy>imgh,gen_box[:,2:5:2]-dx,gen_box[:,2:5:2]+dx)
    for j in range(incs.shape[0]):
      gtimg[i,int(gen_box[j,1]):int(gen_box[j,3]),int(gen_box[j,2]):int(gen_box[j,4]),:] = txt_img[incs[j]]
    gen_box = np.concatenate((boxes,gen_box),axis=0)
    box_list.append(tf.convert_to_tensor(gen_box,dtype=gtbox[i].dtype))
  
  gtimg = tf.convert_to_tensor(gtimg,dtype=img.dtype)
  return gtimg, box_list

# @tf.function
def label_overlap_tf(gt_boxes,org_size,feat_label):
  """
    Args:
      gt_boxes: tensor (gt_num,5) with [label,y1,x1,y2,x2]
        or (gt_num,4)
      org_size: [h,w] of original image
      feat_label: tensor (h,w) with [label]
    Return:
      overlap, tensor with (gt_num) shape
  """
  feat_h,feat_w = feat_label.shape[0]/org_size[0],feat_label.shape[1]/org_size[1]
  # cube_h,cube_w = int(org_size[0]/feat_label.shape[0]),int(org_size[1]/feat_label.shape[1])
  if(gt_boxes.shape[1]==4):
    labels = np.full([gt_boxes.shape[0]],1,dtype=np.int16)
    gt_boxes = tf.stack([gt_boxes[:,0]*feat_h,gt_boxes[:,1]*feat_w,gt_boxes[:,2]*feat_h,gt_boxes[:,3]*feat_w],axis=-1).numpy()
  else:
    labels = gt_boxes[:,0].numpy().astype(np.int16)
    gt_boxes = tf.stack([gt_boxes[:,1]*feat_h,gt_boxes[:,2]*feat_w,gt_boxes[:,3]*feat_h,gt_boxes[:,4]*feat_w],axis=-1).numpy()

  feat_label = feat_label.numpy().astype(np.int16)
  overlaps = []
  for i in range(labels.shape[0]):
    gollabel = np.zeros(feat_label.shape,dtype=np.int16)
    gollabel[math.floor(gt_boxes[i,0]):min(math.ceil(gt_boxes[i,2]),feat_label.shape[0]-1),
      math.floor(gt_boxes[i,1]):min(math.ceil(gt_boxes[i,3]),feat_label.shape[1]-1)] = 1
    gollabel[feat_label==labels[i]] += 1
    if(gollabel[gollabel==1].size>0):
      overlaps+=[gollabel[gollabel==2].size/gollabel[gollabel==1].size]
    else:
      overlaps+=[0]
  return tf.convert_to_tensor(overlaps)

@tf.function
def gen_label_from_prob(probs):
  """
    Args: probs: tensor with shape (h,w,num_class)
      or shape (1,h,w,num_class)
    Return: label tensor with shape (h,w) with tf.int16
      label will be the subscript of highest possible
      and value in [0,num_class-1]
  """
  return tf.reshape(tf.argmax(probs,axis=-1,output_type=tf.int32),probs.shape[-3:-1])

