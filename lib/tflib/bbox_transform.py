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
def feat_layer_cod_gen(org_size,feat_size,class_num=1):
  """
    Generate feature layer coordinate.
    Args:
      org_size: original image size [oh,ow]
      feat_size: feature image size [fh,fw]
      class_num: number of class
    Return:
      tensor with (1,fh,fw,class_num*4) shape
      where 4 is [dy1,dx1,dy2,dx2]
  """
  cube_h,cube_w = float(org_size[0] / feat_size[0]),float(org_size[1] / feat_size[1])
  det_cx,det_cy = tf.meshgrid(
    tf.range(0,feat_size[1]+2,dtype=tf.float32)*cube_w,
    tf.range(0,feat_size[0]+2,dtype=tf.float32)*cube_h)  
  det_mrt = tf.stack([
      det_cy[0:feat_size[0],0:feat_size[1]],
      det_cx[0:feat_size[0],0:feat_size[1]],
      det_cy[1:feat_size[0]+1,1:feat_size[1]+1],
      det_cx[1:feat_size[0]+1,1:feat_size[1]+1],
    ],axis=-1)
  det_mrt = tf.broadcast_to(det_mrt,[class_num,]+det_mrt.shape[-3:-1]+[4])
  det_mrt = tf.transpose(det_mrt,perm=[1,2,0,3])
  return tf.reshape(det_mrt,[1,]+det_mrt.shape[0:2]+[4*class_num])

@tf.function
def xywh2yxyx(boxes):
  """
    Arg: boxes with tensor shape (num_box,5) or (num_box,4)
      where 5 or 4 is ([label] +) [x,y,w,h]
    Return: same shape tensor with ([label] +) [y1,x1,y2,x2]
  """
  if(boxes.shape[-1]==4):
    boxes = tf.stack([boxes[:,-3],boxes[:,-4],boxes[:,-3]+boxes[:,-1],boxes[:,-4]+boxes[:,-2]],axis=1)
  else:
    boxes = tf.stack([boxes[:,0],boxes[:,-3],boxes[:,-4],boxes[:,-3]+boxes[:,-1],boxes[:,-4]+boxes[:,-2]],axis=1)
  return boxes

# @tf.function
def map2coordinate(boxes,org_cod,targ_cod):
  """
    Args:
      boxes: (N,4) with [y1,x1,y2,x2]
      org_cod: original coordinate [height, width]
      targ_cod: target coordinate [height, width]
    Return:
      boxes: (N,4) with [y1,x1,y2,x2]
  """
  fact_x = targ_cod[1]/org_cod[1]
  fact_y = targ_cod[0]/org_cod[0]
  
  return tf.stack([boxes[:,0]*fact_y,boxes[:,1]*fact_x,boxes[:,2]*fact_y,boxes[:,3]*fact_x],axis=1)

def gen_label_from_gt(layer_shape,gt_box,img_shape=None,bg_label=-1):
  """
    Args: 
      layer_shape: layer shape [layer_height, layer_width]
      gt_box: (total_gts,5) with [class, y1, x1, y2, x2]
      img_shape: 
        none with a normalized gt_box coordinate
        or image shape [height, width]
    Return:
      tf.int32 tensor with (layer_height, layer_width)
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
  label_np = np.full(layer_shape,bg_label,np.int32)
  for i in range(gtlabel.shape[0]):
    x1 = max(math.floor(bbox[i,1]),0)
    y1 = max(math.floor(bbox[i,0]),0)
    y2 = min(math.ceil(bbox[i,2]),layer_shape[0])
    x2 = min(math.ceil(bbox[i,3]),layer_shape[1])
    label_np[y1:y2,x1:x2]=gtlabel[i]
  return tf.convert_to_tensor(label_np,dtype=tf.int32)

def gen_label_with_width_from_gt(layer_shape,gt_box,img_shape=None,bg_label=-1):
  """
    Args: 
      layer_shape: layer shape [layer_height, layer_width]
      gt_box: (total_gts,5) with [class, y1, x1, y2, x2]
      img_shape: 
        none with a normalized gt_box coordinate
        or image shape [height, width]
    Return:
      labels: tf.int32 tensor with (layer_height, layer_width)
        where gt will label in [0,numClass-1] and bg is -1
      weights: tf.float32 tensor wirh (layer_height, layer_width)
        for each class, calculate areas / max area as weight
  """
  if(img_shape!=None):
    fact_x = layer_shape[1]/img_shape[1]
    fact_y = layer_shape[0]/img_shape[0]
  else:
    fact_x = layer_shape[1]
    fact_y = layer_shape[0]
  gtlabel = gt_box[:,0].numpy()
  gtlabel = gtlabel.astype(np.int32)
  gtarea = np.zeros((gt_box.shape[0]))
  bbox = tf.stack([gt_box[:,1]*fact_y,gt_box[:,2]*fact_x,gt_box[:,3]*fact_y,gt_box[:,4]*fact_x],axis=1).numpy()
  for i in range(gtlabel.max()+1):
    slc = np.where(gtlabel==i)[0]
    if(slc.shape[0]==0):
      continue
    tmp = (bbox[slc,2]-bbox[slc,0])*(bbox[slc,3]-bbox[slc,1])
    # gtarea[slc] = (tmp-tmp.min())/(tmp.max()-tmp.min())
    gtarea[slc] = tmp/tmp.max()
  label_np = np.full(layer_shape,bg_label,np.int32)
  weight_np = np.full(layer_shape,0.0)
  for i in range(gtlabel.shape[0]):
    x1 = max(math.floor(bbox[i,1]),0)
    y1 = max(math.floor(bbox[i,0]),0)
    y2 = min(math.ceil(bbox[i,2]),layer_shape[0])
    x2 = min(math.ceil(bbox[i,3]),layer_shape[1])
    label_np[y1:y2,x1:x2]=gtlabel[i]
    weight_np[y1:y2,x1:x2]=gtarea[i]
  return tf.convert_to_tensor(label_np,dtype=tf.int32),tf.convert_to_tensor(weight_np,dtype=tf.float32)

def build_boxex_from_path(cls_prb,box_prd,ort_map,target_class,cls_threshold=0.7,dir_threshold=0.7):
  """
    Args:
      cls_prb: tesnor with (1,h,w,num_class), possibilities
      box_prd: tesnor with (1,h,w,(num_class-1)*4)
        where 4 is [y1,x1,y2,x2]
      ort_map: tesnor with (1,h,w,direction)
        where direction=4 is [Up,Left,Down,Right]
        where direction=8 is [Up,upleft,Left,downleft,Down,downright,Right,upright]
    Return:
      list of predicted tesnor with (M,4) where 4 is [y1, x1, y2, x2] in org size
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
  imgh = cls_prb.shape[-3]
  imgw = cls_prb.shape[-2]
  path_list = []
  cluster_count=1
  if(len(box_prd.shape)==4):
    box_prd = tf.reshape(box_prd,box_prd.shape[1:])
  if(len(ort_map.shape)==4):
    ort_map = tf.reshape(ort_map,ort_map.shape[1:])
  if(len(cls_prb.shape)==4):
    cls_prb = tf.reshape(cls_prb,cls_prb.shape[1:])
  cls_prb = cls_prb.numpy()
  incs = tf.where(cls_prb[:,:,target_class]>cls_threshold).numpy()
  mask = np.full((imgh,imgw),-1,dtype=np.int16)
  mask[cls_prb[:,:,target_class]>cls_threshold] = 0
  ort_map = ort_map.numpy()
  for inc in incs:
    dirps = ort_map[inc[0],inc[1]]
    dpoint = np.where(dirps>dir_threshold)[0]
    if(dpoint.shape[0]==0):
      continue
    my_label = mask[inc[0],inc[1]]
    if(my_label!=0):
      # ONLY point itself can add box to list
      path_list[my_label-1]=tf.concat([path_list[my_label-1],tf.reshape(box_prd[inc[0],inc[1],:],[1,4])],axis=0)

    for dpt in dpoint:
      dy,dx = direction[dpt]
      if((inc[0]+dy)<0 or (inc[0]+dy)>=imgh or (inc[1]+dx)<0 or (inc[1]+dx)>=imgw):
        continue
      if(my_label!=0 and mask[inc[0]+dy,inc[1]+dx]==0):
        # my_label==label, label neighbor
        mask[inc[0]+dy,inc[1]+dx] = my_label

      elif(mask[inc[0]+dy,inc[1]+dx]>0):
        # my_label==0, find one neighbor
        mask[inc[0],inc[1]] = mask[inc[0]+dy,inc[1]+dx]
        my_label = mask[inc[0]+dy,inc[1]+dx]
        path_list[my_label-1]=tf.concat([path_list[my_label-1],tf.reshape(box_prd[inc[0],inc[1],:],[1,4])],axis=0)
        break

    if(my_label==0):
      # father
      mask[inc[0],inc[1]]=cluster_count
      my_label=cluster_count
      path_list+=[tf.reshape(box_prd[inc[0],inc[1],:],[1,4])]
      for dpt in dpoint:
        dy,dx = direction[dpt]
        if((inc[0]+dy)<0 or (inc[0]+dy)>=imgh or (inc[1]+dx)<0 or (inc[1]+dx)>=imgw):
          continue
        # label all neighbor
        if(mask[inc[0]+dy,inc[1]+dx]==0):
          mask[inc[0]+dy,inc[1]+dx]=my_label
      cluster_count+=1
  # mask = tf.convert_to_tensor(mask,dtype=tf.int32)
  return path_list, mask

def get_label_from_mask(gt_box,mask):
  """
    Args:
      gt_box: tensor (total_gts,4) with [y1, x1, y2, x2]
      mask: (height,width) with [label]
    return:
      list [total_gts] with the highest label
  """
  gt_box = gt_box.numpy()
  gt_label_list = []
  for box in gt_box:
    submk = mask[math.floor(box[0]):math.ceil(box[2]),math.floor(box[1]):math.ceil(box[3])].reshape([-1])
    inc = np.argmax(np.bincount(submk))
    gt_label_list += [inc]
  return gt_label_list

def pre_box_loss(gt_box, pred_map, org_size=None, sigma=1.0):
  """
    Args:
      gt_box: tensor (total_gts,4) with [y1, x1, y2, x2]
        in original coordinate.
      pred_map: predict layer tensor (h,w,4) with [y1, x1, y2, x2]
      org_size: original image size (h,w)
    Return:
      loss value
  """
  sigma_2 = sigma ** 2
  pred_map = tf.reshape(pred_map,pred_map.shape[-3:])

  if(org_size==None):
    feat_h,feat_w = 1,1
    cube_h,cube_w = 1,1
  else:
    feat_h,feat_w = pred_map.shape[-3]/org_size[0],pred_map.shape[-2]/org_size[1]
    cube_h,cube_w = org_size[0]/pred_map.shape[-3],org_size[1]/pred_map.shape[-2]
    
  gt_box = gt_box.numpy()
  abs_boundary_det = []
  abs_inside_det = []
  for box in gt_box:
    box_h,box_w = box[2]-box[0],box[3]-box[1]
    # y_start,y_end = math.floor(box[0]*feat_h),min(math.ceil(box[2]*feat_h),pred_map.shape[-3]-1)
    # x_start,x_end = math.floor(box[1]*feat_w),min(math.ceil(box[3]*feat_w),pred_map.shape[-2]-1)
    x_start,x_end = box[1] / cube_w,box[3] / cube_w
    y_start,y_end = box[0] / cube_h,box[2] / cube_h
    x_start = math.floor(x_start) if(x_start - int(x_start)) < 0.8 else math.ceil(x_start)
    y_start = math.floor(y_start) if(y_start - int(y_start)) < 0.8 else math.ceil(y_start)
    x_end = math.floor(x_end) if(x_end - int(x_end)) < 0.2 else math.ceil(x_end)
    y_end = math.floor(y_end) if(y_end - int(y_end)) < 0.2 else math.ceil(y_end)

    tmp_tag = cube_h*(y_start+1) if ((y_end-1)>y_start) else box[2]
    tmp_ys = tf.stack([
      # up boundary
      tf.math.abs(pred_map[y_start,x_start:x_end,0] - box[0]),
      tf.math.abs(pred_map[y_start,x_start:x_end,2] - tmp_tag),
    ],axis=-1)
    tmp_ys = tf.where(tmp_ys > (1. / sigma_2),tmp_ys - (0.5 / sigma_2),tf.pow(tmp_ys, 2) * (sigma_2 / 2.))
    tmp_ys = tf.reduce_sum(tmp_ys,axis=-1)
    tmp_ye = 0.0
    if((y_end-1)>y_start):
      tmp_ye = tf.stack([
        # down boundary
        tf.math.abs(pred_map[(y_end-1),x_start:x_end,0] - cube_h*(y_end-1)),
        tf.math.abs(pred_map[(y_end-1),x_start:x_end,2] - box[2]),
      ],axis=-1)
      tmp_ye = tf.where(tmp_ye > (1. / sigma_2),tmp_ye - (0.5 / sigma_2),tf.pow(tmp_ye, 2) * (sigma_2 / 2.))
      tmp_ye = tf.reduce_sum(tmp_ye,axis=-1)
    tmp_tag = cube_w*(x_start+1) if ((x_end-1)>x_start) else box[3]
    tmp_xs = tf.stack([
      # down boundary
      tf.math.abs(pred_map[y_start:y_end,x_start,1] - box[1]),
      tf.math.abs(pred_map[y_start:y_end,x_start,3] - tmp_tag),
    ],axis=-1)
    tmp_xs = tf.where(tmp_xs > (1. / sigma_2),tmp_xs - (0.5 / sigma_2),tf.pow(tmp_xs, 2) * (sigma_2 / 2.))
    tmp_xs = tf.reduce_sum(tmp_xs,axis=-1)
    tmp_xe = 0.0
    if((x_end-1)>x_start):
      tmp_xe = tf.stack([
        # down boundary
        tf.math.abs(pred_map[y_start:y_end,(x_end-1),3] - cube_w*(x_end-1)),
        tf.math.abs(pred_map[y_start:y_end,(x_end-1),3] - box[3]),
      ],axis=-1)
      tmp_xe = tf.where(tmp_xe > (1. / sigma_2),tmp_xe - (0.5 / sigma_2),tf.pow(tmp_xe, 2) * (sigma_2 / 2.))
      tmp_xe = tf.reduce_sum(tmp_xe,axis=-1)
    tmp_det = tf.math.reduce_mean(tmp_ys)/box_h+tf.math.reduce_mean(tmp_ye)/box_h+tf.math.reduce_mean(tmp_xs)/box_w+tf.math.reduce_mean(tmp_xe)/box_w

    y_start+=1
    y_end-=1
    x_start+=1
    x_end-=1
    tmp_in_det = 0.0
    # if(y_start<y_end and x_start<x_end):
    if(False):
      # tmp_ys = tf.math.abs(pred_map[y_start:y_end,x_start:x_end,0] - det_cy[y_start:y_end,x_start:x_end])
      tmp_ys = tf.where(pred_map[y_start:y_end,x_start:x_end,0]<box[0])
      if(tmp_ys.shape[0]>0):
        tmp_ys = tf.math.abs(tf.gather_nd(pred_map[y_start:y_end,x_start:x_end,0],tmp_ys) - box[0])
        tmp_ys = tf.where(tmp_ys > (1. / sigma_2),tmp_ys - (0.5 / sigma_2),tf.pow(tmp_ys, 2) * (sigma_2 / 2.))
        tmp_in_det += tf.math.reduce_mean(tmp_ys)/box_h

      # tmp_ye = tf.math.abs(pred_map[y_start:y_end,x_start:x_end,2] - det_cy[(y_start+1):(y_end+1),(x_start+1):(x_end+1)])
      tmp_ye = tf.where(pred_map[y_start:y_end,x_start:x_end,2]>box[2])
      if(tmp_ye.shape[0]>0):
        tmp_ye = tf.math.abs(tf.gather_nd(pred_map[y_start:y_end,x_start:x_end,2],tmp_ye) - box[2])
        tmp_ye = tf.where(tmp_ye > (1. / sigma_2),tmp_ye - (0.5 / sigma_2),tf.pow(tmp_ye, 2) * (sigma_2 / 2.))
        tmp_in_det += tf.math.reduce_mean(tmp_ye)/box_h

      # tmp_xs = tf.math.abs(pred_map[y_start:y_end,x_start:x_end,1] - det_cx[y_start:y_end,x_start:x_end])
      tmp_xs = tf.where(pred_map[y_start:y_end,x_start:x_end,1]<box[1])
      if(tmp_xs.shape[0]>0):
        tmp_xs = tf.math.abs(tf.gather_nd(pred_map[y_start:y_end,x_start:x_end,1],tmp_xs) - box[1])
        tmp_xs = tf.where(tmp_xs > (1. / sigma_2),tmp_xs - (0.5 / sigma_2),tf.pow(tmp_xs, 2) * (sigma_2 / 2.))
        tmp_in_det += tf.math.reduce_mean(tmp_xs)/box_w

      # tmp_xe = tf.math.abs(pred_map[y_start:y_end,x_start:x_end,3] - det_cx[(y_start+1):(y_end+1),(x_start+1):(x_end+1)])
      tmp_xe = tf.where(pred_map[y_start:y_end,x_start:x_end,3]>box[3])
      if(tmp_xe.shape[0]>0):
        tmp_xe = tf.math.abs(tf.gather_nd(pred_map[y_start:y_end,x_start:x_end,3],tmp_xe) - box[3])
        tmp_xe = tf.where(tmp_xe > (1. / sigma_2),tmp_xe - (0.5 / sigma_2),tf.pow(tmp_xe, 2) * (sigma_2 / 2.))
        tmp_in_det += tf.math.reduce_mean(tmp_xe)/box_w

      # tmp_y = tf.math.abs(pred_map[y_start,x_start:x_end,0] - det_cy[y_start:y_end,x_start:x_end]) \
      #   + tf.math.abs(pred_map[y_end,x_start:x_end,2] - det_cy[(y_start+1):(y_end+1),(x_start+1):(x_end+1)])
      # tmp_y = tf.where(tmp_y > (1. / sigma_2),tmp_y - (0.5 / sigma_2),tf.pow(tmp_y, 2) * (sigma_2 / 2.))
      # tmp_x = tf.math.abs(pred_map[y_start,x_start:x_end,1] - det_cx[y_start:y_end,x_start:x_end]) \
      #   + tf.math.abs(pred_map[y_end,x_start:x_end,3] - det_cx[(y_start+1):(y_end+1),(x_start+1):(x_end+1)])
      # tmp_x = tf.where(tmp_x > (1. / sigma_2),tmp_x - (0.5 / sigma_2),tf.pow(tmp_x, 2) * (sigma_2 / 2.))
      # tmp_det += tf.math.reduce_sum(tmp_y)+tf.math.reduce_sum(tmp_x)
    abs_boundary_det+=[tmp_det]
    abs_inside_det+=[tmp_in_det]
  return tf.convert_to_tensor(abs_boundary_det)

