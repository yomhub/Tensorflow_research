import tensorflow as tf
import numpy as np
import math

@tf.function
def bbox_transform(ex_rois, gt_rois):
  """
  Args:
    ex_rois: Tensor with shape (N,4)
    gt_rois: Tensor with shape (N,4)
      where 4 is [y1,x1,y2,x2]
    
  Assert:
    ex_rois and gt_rois have same shape
    
  Return:
    difference of bounding-box 
    with shape (N,4)
    where 4 is [dx,dy,dw,dh]
  """
  assert ex_rois.shape[0] == gt_rois.shape[0]

  ex_heights = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
  ex_widths = ex_rois[:, 3] - ex_rois[:, 1] + 1.0 
  ex_ctr_y = ex_rois[:, 0] + 0.5 * ex_widths
  ex_ctr_x = ex_rois[:, 1] + 0.5 * ex_heights

  gt_heights = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
  gt_widths = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
  gt_ctr_y = gt_rois[:, 0] + 0.5 * gt_widths
  gt_ctr_x = gt_rois[:, 1] + 0.5 * gt_heights

  targets_dx = tf.clip_by_value((gt_ctr_x - ex_ctr_x) / ex_widths,1e-8,1000.0)
  targets_dy = tf.clip_by_value((gt_ctr_y - ex_ctr_y) / ex_heights,1e-8,1000.0)
  targets_dw = tf.math.log(tf.clip_by_value(gt_widths / ex_widths,1e-8,1000.0))
  targets_dh = tf.math.log(tf.clip_by_value(gt_heights / ex_heights,1e-8,1000.0))

  targets = tf.stack(
    [targets_dx, targets_dy, targets_dw, targets_dh],1)
  
  return targets

@tf.function
def bbox_transform_inv_tf(boxes, deltas, order='xyxy'):
  """
    Args:
      boxes: boxes (N,4) 
      with order 'xyxy' [x1,y1,x2,y2]
      or order 'yxyx' [y1,x1,y2,x2]
      deltas: deltas value (N,4)
      
  """
  boxes = tf.cast(boxes, deltas.dtype)
  # if(order=='xyxy'):
  widths = tf.subtract(boxes[:, 2], boxes[:, 0]) + 1.0
  heights = tf.subtract(boxes[:, 3], boxes[:, 1]) + 1.0
  ctr_x = tf.add(boxes[:, 0], widths * 0.5)
  ctr_y = tf.add(boxes[:, 1], heights * 0.5)
  # else:
  #   heights = tf.subtract(boxes[:, 2], boxes[:, 0]) + 1.0
  #   widths = tf.subtract(boxes[:, 3], boxes[:, 1]) + 1.0
  #   ctr_y = tf.add(boxes[:, 0], widths * 0.5)
  #   ctr_x = tf.add(boxes[:, 1], heights * 0.5)

  dx = deltas[:, 0]
  dy = deltas[:, 1]
  dw = deltas[:, 2]
  dh = deltas[:, 3]

  pred_ctr_x = tf.add(tf.multiply(dx, widths), ctr_x)
  pred_ctr_y = tf.add(tf.multiply(dy, heights), ctr_y)
  pred_w = tf.multiply(tf.exp(dw), widths)
  pred_h = tf.multiply(tf.exp(dh), heights)

  pred_boxes0 = tf.subtract(pred_ctr_x, pred_w * 0.5)
  pred_boxes1 = tf.subtract(pred_ctr_y, pred_h * 0.5)
  pred_boxes2 = tf.add(pred_ctr_x, pred_w * 0.5)
  pred_boxes3 = tf.add(pred_ctr_y, pred_h * 0.5)

  # # if(maxx!=None):
  # pred_boxes0 = tf.clip_by_value(pred_boxes0, 0, im_info[1] - 1)
  # pred_boxes2 = tf.clip_by_value(pred_boxes2, 0, im_info[1] - 1)

  # # if(maxy!=None):
  # pred_boxes1 = tf.clip_by_value(pred_boxes1, 0, im_info[0] - 1)
  # pred_boxes3 = tf.clip_by_value(pred_boxes3, 0, im_info[0] - 1)

  return tf.stack([pred_boxes1, pred_boxes0, pred_boxes3, pred_boxes2], axis=1)

# @tf.function
def clip_boxes_tf(boxes, im_info):
  # boxes=[y1,x1,y2,x2] im_info=[y,x] out=[y1,x1,y2,x2]
  b02 = tf.clip_by_value(boxes[:, 0:3:2], clip_value_min=0, clip_value_max=im_info[0] - 1)
  b13 = tf.clip_by_value(boxes[:, 1:4:2], clip_value_min=0, clip_value_max=im_info[1] - 1)
  return tf.stack([b02[:,0], b13[:,0], b02[:,1], b13[:,1]], axis=1)

@tf.function
def xywh2yxyx(boxes):
  return tf.stack([boxes[:,1],boxes[:,0],boxes[:,1]+boxes[:,3],boxes[:,0]+boxes[:,2]],axis=1)

# @tf.function
def map2coordinate(boxes,org_cod,targ_cod):
  """
    Args:
      boxes: (N,4) with [y1,x1,y2,x2]
      org_cod: original coordinate [height，width]
      targ_cod: target coordinate [height，width]
    Return:
      boxes: (N,4) with [y1,x1,y2,x2]
  """
  fact_x = targ_cod[1]/org_cod[1]
  fact_y = targ_cod[0]/org_cod[0]
  
  return tf.stack([boxes[:,0]*fact_y,boxes[:,1]*fact_x,boxes[:,2]*fact_y,boxes[:,3]*fact_x],axis=1)

def label_layer(layer_shape,gt_box,img_shape=None):
  """
    Args: 
      layer_shape: layer shape [layer_height，layer_width]
      gt_box: (total_gts,5) with [class, y1, x1, y2, x2]
      img_shape: 
        none with a normalized gt_box coordinate
        or image shape [height，width]
    Return:
      tf.int32 tensor with (layer_height，layer_width)
      where gt will label in [0,numClass-1]
      and bg is -1
  """
  if(img_shape!=None):
    fact_x = layer_shape[1]/img_shape[1]
    fact_y = layer_shape[0]/img_shape[0]
  else:
    fact_x = layer_shape[1]
    fact_y = layer_shape[0]
  gtlabel = gt_box[:,0].numpy()
  gtlabel = gtlabel.astype(np.int32)
  bbox = tf.stack([gt_box[:,1]*fact_y,gt_box[:,2]*fact_x,gt_box[:,3]*fact_y,gt_box[:,4]*fact_x],axis=1).numpy()
  label_np = np.full(layer_shape,-1,np.int32)
  for i in range(gtlabel.shape[0]):
    x1 = max(math.floor(bbox[i,1]),0)
    y1 = max(math.floor(bbox[i,0]),0)
    y2 = min(math.ceil(bbox[i,2]),layer_shape[0])
    x2 = min(math.ceil(bbox[i,3]),layer_shape[1])
    label_np[y1:y2,x1:x2]=gtlabel[i]
  return tf.convert_to_tensor(label_np,dtype=tf.int32)
  
def build_boxex_from_path(cls_prb,box_prd,ort_map,target_class,threshold=0.5):
  """
    Args:
      cls_prb: tesnor with (1,h,w,num_class)
      box_prd: tesnor with (1,h,w,(num_class-1)*4)
        where 4 is [y1,x1,y2,x2]
      ort_map: tesnor with (1,h,w,direction)
        where direction=4 is [Up,Left,Down,Right]
        where direction=8 is [Up,upleft,Left,downleft,Down,downright,Right,upright]
    Return:
      list of p[]
  """
  direction = ort_map.shape[-1]
  if(ort_map.shape[-1]==8):
    direction={
      # [dy,dx], Up,upleft,Left,downleft
      0:[-1,0],1:[-1,-1],2:[0,-1],3:[1,-1],
      # Down,downright,Right,upright
      4:[1,0],5:[1,1],6:[0,1],7:[-1,1],
    }
  else:
    direction={
      # [dy,dx], Up,Left,Down,Right
      0:[-1,0],1:[0,-1],2:[1,0],3:[0,1],
    }
  
  path_list = []
  path_label_list = []
  cls_prb = tf.reshape(cls_prb,cls_prb.shape[1:]).numpy()
  incs = tf.where(cls_prb[:,:,target_class]>threshold).numpy()
  mask = np.full((cls_prb.shape[-3],cls_prb.shape[-2]),-1,dtype=np.int16)
  mask = np.select(cls_prb[:,:,target_class]>threshold,0,-1)

  ort_map = ort_map.numpy()
  for inc in incs:
    dirps = ort_map[inc[0],inc[1]]
    dpoint = np.where(dirps==dirps.max)
    dy,dx = direction[dpoint]
    mask

  return path_list