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
    where 4 is [dx,dy,log(dw/w),log(dh/h)]
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
    Apply deltas to anchors, 
      x+=dx*widths,y+=dy*heights
      w+=exp(dw),h+=exp(dh)
    Args:
      boxes: boxes (N,[y1,x1,y2,x2]) in 'yxyx'
                or (N,[x1,y1,x2,y2]) in 'xyxy'
      deltas: deltas value (N,[dy,dx,dh,dw]) in 'yxyx'
                        or (N,[dx,dy,dw,dh]) in 'xyxy'
    Returns:
      Rectified box (N,[x1,y1,x2,y2]) in 'xyxy'
                or  (N,[y1,x1,y2,x2]) in 'yxyx'
  """
  order=order.lower()
  boxes = tf.cast(boxes, deltas.dtype)
  x_start = 1 if order=='yxyx' else 0
  y_start = 0 if order=='yxyx' else 1

  widths = tf.subtract(boxes[:, x_start+2], boxes[:, x_start]) + 1.0
  heights = tf.subtract(boxes[:, y_start+2], boxes[:, y_start]) + 1.0
  ctr_x = tf.add(boxes[:, x_start], widths * 0.5)
  ctr_y = tf.add(boxes[:, y_start], heights * 0.5)

  dx = deltas[:, x_start]
  dy = deltas[:, y_start]
  dw = deltas[:, x_start+2]
  dh = deltas[:, y_start+2]

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
  if(order=='yxyx'):
    return tf.stack([pred_boxes1, pred_boxes0, pred_boxes3, pred_boxes2], axis=1)
  return tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)

# @tf.function
def clip_boxes_tf(boxes, im_info, order='xyxy'):
  """
    Clip box into image.
    Args:
      boxes: boxes (N,[y1,x1,y2,x2]) in 'yxyx'
                or (N,[x1,y1,x2,y2]) in 'xyxy'
      im_info: [h,w]
    Return: Cliped boxes (N,[y1,x1,y2,x2])
  """
  # boxes=[y1,x1,y2,x2] im_info=[y,x] out=[y1,x1,y2,x2]
  order=order.lower()
  x_start = 1 if order=='yxyx' else 0
  y_start = 0 if order=='yxyx' else 1
  # y
  b02 = tf.clip_by_value(boxes[:, y_start::2], clip_value_min=0, clip_value_max=im_info[0] - 1)
  # x
  b13 = tf.clip_by_value(boxes[:, x_start::2], clip_value_min=0, clip_value_max=im_info[1] - 1)
  
  return tf.stack([b02[:,0], b13[:,0], b02[:,1], b13[:,1]], axis=1)

@tf.function
def feat_layer_cod_gen(recf_size,feat_size,class_num=1):
  """
    Generate feature layer bias coordinate in orign image.
    Args:
      recf_size: receptive field size [rh,rw,sy,sx]
        where rh,rw is receptive field size
        and sy,sh is stride size in y,x
      feat_size: feature image size [fh,fw]
      class_num: number of class
    Return:
      tensor with (1,fh,fw,class_num*4) shape
      where 4 is [dy1,dx1,dy2,dx2]
  """
  cube_h,cube_w,stride_y,stride_x = recf_size[0],recf_size[1],recf_size[2],recf_size[3]
  stpx2 = cube_w/2 + stride_x/2
  stpy2 = cube_h/2 + stride_y/2
  det_cx,det_cy = tf.meshgrid(
    tf.range(0,int(feat_size[1]+2),dtype=tf.float32)*stride_x,
    tf.range(0,int(feat_size[0]+2),dtype=tf.float32)*stride_y
  )
  det_mrt = tf.stack([
      det_cy[0:feat_size[0],0:feat_size[1]]+stpy2-cube_h,
      det_cx[0:feat_size[0],0:feat_size[1]]+stpx2-cube_w,
      det_cy[0:feat_size[0],0:feat_size[1]]+stpy2,
      det_cx[0:feat_size[0],0:feat_size[1]]+stpx2,
    ],axis=-1)
  det_mrt = tf.clip_by_value(
    det_mrt,0.0,
    tf.maximum(float(feat_size[0])*stride_y+cube_h,float(feat_size[1])*stride_x+cube_w))
  det_mrt = tf.broadcast_to(det_mrt,[class_num,]+det_mrt.shape[-3:-1]+[4])
  det_mrt = tf.transpose(det_mrt,perm=[1,2,0,3])
  return tf.reshape(det_mrt,[1,]+det_mrt.shape[0:2]+[4*class_num])

@tf.function
def xywh2yxyx(boxes:tf.Tensor, center:bool=False) -> tf.Tensor:
  """
    Arg: boxes with tensor shape (num_box,5) or (num_box,4)
      where 5 or 4 is ([label] +) [x,y,w,h]
    Return: same shape tensor with ([label] +) [y1,x1,y2,x2]
  """
  if(boxes.shape[-1]==4):
    if(center):
      boxes = tf.stack([boxes[:,-3]-boxes[:,-1]/2.0,boxes[:,-4]-boxes[:,-2]/2.0,boxes[:,-3]+boxes[:,-1]/2.0,boxes[:,-4]+boxes[:,-2]/2.0],axis=1)
    else:
      boxes = tf.stack([boxes[:,-3],boxes[:,-4],boxes[:,-3]+boxes[:,-1],boxes[:,-4]+boxes[:,-2]],axis=1)
  else:
    if(center):
      boxes = tf.stack([boxes[:,0],boxes[:,-3]-boxes[:,-1]/2.0,boxes[:,-4]-boxes[:,-2]/2.0,boxes[:,-3]+boxes[:,-1]/2.0,boxes[:,-4]+boxes[:,-2]/2.0],axis=1)
    else:
      boxes = tf.stack([boxes[:,0],boxes[:,-3],boxes[:,-4],boxes[:,-3]+boxes[:,-1],boxes[:,-4]+boxes[:,-2]],axis=1)
  return boxes

@tf.function
def yxyx2xywh(boxes:tf.Tensor, center:bool=False) -> tf.Tensor:
  """
  Convert [y1,x1,y2,x2] to [cx,cy,w,h]
    Arg: 
      boxes with tensor shape (num_box,5) or (num_box,4)
        where 5 or 4 is ([label] +) [y1,x1,y2,x2]
      center: True to locate [cx,cy] in center of box
        False to locate [cx,cy] in topleft of box
    Return: same shape tensor with ([label] +) [cx,cy,w,h]
  """
  ws = tf.math.abs(boxes[:,-1]-boxes[:,-3])
  hs = tf.math.abs(boxes[:,-2]-boxes[:,-4])
  cxs = boxes[:,-3]+ws/2.0 if(center)else boxes[:,-3]
  cys = boxes[:,-4]+hs/2.0 if(center)else boxes[:,-4]

  if(boxes.shape[-1]==4):
    boxes = tf.stack([cxs,cys,ws,hs],axis=1)
  else:
    boxes = tf.stack([boxes[:,0],cxs,cys,ws,hs],axis=1)
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
  if(boxes.shape[-1]==5):
    return tf.stack([boxes[:,0],boxes[:,-4]*fact_y,boxes[:,-3]*fact_x,boxes[:,-2]*fact_y,boxes[:,-1]*fact_x],axis=1)
  else:
    return tf.stack([boxes[:,-4]*fact_y,boxes[:,-3]*fact_x,boxes[:,-2]*fact_y,boxes[:,-1]*fact_x],axis=1)

def gen_label_from_gt(layer_shape,recf_size,gt_box,img_shape,bg_label=-1):
  """
    Generate label mask from gtbox
    Args: 
      layer_shape: layer shape [layer_height, layer_width]
      recf_size: 
        receptive field size (rf_h,rf_w,stride_y,stride_x)
        or none for center point model.
      gt_box: (total_gts,5) with [class, y1, x1, y2, x2]
      img_shape: 
        none with a normalized gt_box coordinate
        or image shape [height, width] with gt_box in original coordinate
    Return:
      label mask:
        tf.int32 tensor with shape (layer_height, layer_width)
        where gt will label in [0,numClass-1]
        and bg is -1
  """  
  gtlabel = gt_box[:,0].numpy()
  gtlabel = gtlabel.astype(np.int32)
  # init mask as bg label
  label_np = np.full(layer_shape,bg_label,np.int32)

  if(recf_size!=None):
    # label pixels inside receptive field
    cube_h,cube_w,stride_y,stride_x = float(recf_size[0]),float(recf_size[1]),float(recf_size[2]),float(recf_size[3])
    bbox = gt_box[:,1:]
    for i in range(gtlabel.shape[0]):
      x_start,x_end = bbox[i,1] / stride_x,(bbox[i,3]-cube_w) / stride_x
      y_start,y_end = bbox[i,0] / stride_y,(bbox[i,2]-cube_h) / stride_y
      x_start = math.floor(x_start) if(x_start - int(x_start)) < 0.7 else math.ceil(x_start)
      y_start = math.floor(y_start) if(y_start - int(y_start)) < 0.7 else math.ceil(y_start)
      x_end = math.floor(x_end) if(x_end - int(x_end)) < 0.2 else math.ceil(x_end)
      y_end = math.floor(y_end) if(y_end - int(y_end)) < 0.2 else math.ceil(y_end)
      x_start = max(x_start,0)
      y_start = max(y_start,0)
      y_end = min(y_end,layer_shape[0])
      x_end = min(x_end,layer_shape[1])
      label_np[y_start:y_end,x_start:x_end]=gtlabel[i]
  else:
    # label pixel in center of box
    if(img_shape!=None):
      stride_y,stride_x = img_shape[0]/layer_shape[0],img_shape[1]/layer_shape[1]
    else:
      stride_y,stride_x = 1,1
      
    bbox = yxyx2xywh(gt_box[:,1:])
    csxs,cexs = tf.math.floor(bbox[:,0]).numpy(),tf.math.ceil(bbox[:,0]).numpy()+1
    csys,ceys = tf.math.floor(bbox[:,1]).numpy(),tf.math.ceil(bbox[:,1]).numpy()+1
    for i in range(gtlabel.shape[0]):
      label_np[csys[i]:y_end,ceys[i]:cexs[i]]=gtlabel[i]

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
    Build connect boxes into box list.
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
      pred_map: coordinate in prediction layer 
        tensor ((1),h,w,4) with [y1, x1, y2, x2]
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
    x_start = math.floor(x_start) if(x_start - int(x_start)) < 0.7 else math.ceil(x_start)
    y_start = math.floor(y_start) if(y_start - int(y_start)) < 0.7 else math.ceil(y_start)
    x_end = math.floor(x_end) if(x_end - int(x_end)) < 0.2 else math.ceil(x_end)
    y_end = math.floor(y_end) if(y_end - int(y_end)) < 0.2 else math.ceil(y_end)

    tmp_tag = cube_h*(y_start+1) if ((y_end-1)>y_start) else box[2]
    cube_det = tf.range((x_start+1),x_end,dtype=tf.float32)*box_w
    tmp_ys = tf.stack([
      # up boundary
      tf.math.abs(pred_map[y_start,x_start:x_end,0] - box[0]),
      # tf.math.abs(pred_map[y_start,x_start:x_end,1] - tf.concat([box[1],cube_det],axis=0)),
      tf.math.abs(pred_map[y_start,x_start:x_end,2] - tmp_tag),
      # tf.math.abs(pred_map[y_start,x_start:x_end,3] - tf.concat([cube_det,box[3]],axis=0)),
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
    cube_det = tf.range((y_start+1),y_end,dtype=tf.float32)*box_h
    tmp_xs = tf.stack([
      # down boundary
      # tf.math.abs(pred_map[y_start:y_end,x_start,0] - tf.concat([box[0],cube_det],axis=0)),
      tf.math.abs(pred_map[y_start:y_end,x_start,1] - box[1]),
      # tf.math.abs(pred_map[y_start:y_end,x_start,2] - tf.concat([cube_det,box[2]],axis=0)),
      tf.math.abs(pred_map[y_start:y_end,x_start,3] - tmp_tag),
    ],axis=-1)
    tmp_xs = tf.where(tmp_xs > (1. / sigma_2),tmp_xs - (0.5 / sigma_2),tf.pow(tmp_xs, 2) * (sigma_2 / 2.))
    tmp_xs = tf.reduce_sum(tmp_xs,axis=-1)

    tmp_xe = 0.0
    if((x_end-1)>x_start):
      tmp_xe = tf.stack([
        # down boundary
        tf.math.abs(pred_map[y_start:y_end,(x_end-1),1] - cube_w*(x_end-1)),
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


def pre_box_loss_by_det(gt_box, det_map, recf_size, sigma=1.0, use_cross=True, mag_f='smooth', bdf=0.2):
  """
    Args:
      gt_box: tensor (total_gts,4) with [y1, x1, y2, x2]
        in original coordinate.
      det_map: deta in prediction layer 
        tensor ((1),h,w,4) with [dy1, dx1, dy2, dx2] in pixel unit
      recf_size: receptive field size (rf_h,rf_w,stride_y,stride_x)
      mag_f: magnification function, can be
        string: 'smooth', apply L1 smooth on loss function 
        string: 'sigmoid', apply sigmoid on loss function 
        tf.keras.layers.Lambda: apply input Lambda object
        others, don't apply any magnification function
      bdf: boundary for flexible interval, 
        boundary in (1-bdf,1) will be conseiered into next box 
        boundary in (0,bdf) will be conseiered into previous box 
    Return:
      loss value
  """
  pred_map = tf.reshape(det_map,det_map.shape[-3:])
  use_cross=False
  cube_h,cube_w,stride_y,stride_x = float(recf_size[0]),float(recf_size[1]),float(recf_size[2]),float(recf_size[3])
  feat_h,feat_w = det_map.shape[-3],det_map.shape[-2]
  if(bdf>0.5 or bdf<0.0):bdf=0.2
  if(type(mag_f)==str):
    mag_f = mag_f.lower()
    if(mag_f=='tanh'):
      mag_f = tf.keras.layers.Lambda(lambda x: tf.math.sigmoid(x))
    elif(mag_f=='smooth'):
      sigma_2 = sigma**2
      mag_f = tf.keras.layers.Lambda(lambda x: tf.where(x > (1. / x),x - (0.5 / sigma_2),tf.pow(x, 2) * (sigma_2 / 2.)))
    else:
      mag_f = tf.keras.layers.Lambda(lambda x:x)
  elif(mag_f == None):
    mag_f = tf.keras.layers.Lambda(lambda x:x)
  # No need to convert type tf.keras.layers.Lambda

  abs_boundary_det = []
  for box in gt_box:
    f_x_start,f_x_end = box[1] / stride_x,(box[3]-cube_w) / stride_x
    f_y_start,f_y_end = box[0] / stride_y,(box[2]-cube_h) / stride_y
    x_start = math.floor(f_x_start) if(f_x_start - int(f_x_start)) < (1-bdf) else math.ceil(f_x_start)
    y_start = math.floor(f_y_start) if(f_y_start - int(f_y_start)) < (1-bdf) else math.ceil(f_y_start)
    x_end = math.floor(f_x_end) if(f_x_end - int(f_x_end)) < bdf else math.ceil(f_x_end)
    y_end = math.floor(f_y_end) if(f_y_end - int(f_y_end)) < bdf else math.ceil(f_y_end)
    if(x_end<=x_start):x_end=x_start+1
    elif(x_end>=feat_w):x_end=feat_w-1
    if(y_end<=y_start):y_end=y_start+1
    elif(y_end>=feat_h):y_end=feat_h-1

    tmp_det_sum = 0.0
    tmp_tag = 0.0 if (y_end-1)>y_start else cube_h - (box[2]-float(y_start*stride_y))
    # Top boundary
    tmp_det_sum += tf.reduce_mean(mag_f(tf.math.abs(box[0] - float(y_start*stride_y) - pred_map[y_start,x_start:x_end,0]))) # 0, up side
    tmp_det_sum += tf.reduce_mean(mag_f(tf.math.abs(pred_map[y_start,x_start:x_end,2] - tmp_tag))) # 2, down side
    if(use_cross and x_end-x_start>3):
      tmp_det_sum += tf.reduce_mean(mag_f(tf.math.abs(pred_map[y_start,(x_start+1):(x_end-1),3]))) # 3, right side
      tmp_det_sum += tf.reduce_mean(mag_f(tf.math.abs(pred_map[y_start,(x_start+1):(x_end-1),1]))) # 1, left side

    # Bottom boundary, y_end-1 means use Y2 in previous box
    # (yend-1)       gty2  yend
    # |---------------x----|--------------------|
    # gty2 - yend * stride_y < 0
    # (yend-1)           yend gty2
    # |--------------------|-x------------------|
    # gty2 - yend * stride_y > 0
    if((y_end-1)>y_start):
      tmp_det_sum += tf.reduce_mean(mag_f(tf.math.abs(pred_map[(y_end-1),x_start:x_end,0])))
      tmp_det_sum += tf.reduce_mean(mag_f(tf.math.abs(box[2] - float(y_end*stride_y) - pred_map[(y_end-1),x_start:x_end,2])))
      if(use_cross and x_end-x_start>3):
        tmp_det_sum += tf.reduce_mean(mag_f(tf.math.abs(pred_map[(y_end-1),(x_start+1):(x_end-1),1])))
        tmp_det_sum += tf.reduce_mean(mag_f(tf.math.abs(pred_map[(y_end-1),(x_start+1):(x_end-1),3])))
    
    # Left boundary
    tmp_tag = 0.0 if ((x_end-1)>x_start) else cube_w - (box[3]-float(x_start*stride_x))
    tmp_det_sum += tf.reduce_mean(mag_f(tf.math.abs(box[1] - float(x_start*stride_x)  - pred_map[y_start:y_end,x_start,1]))) # 1, left side
    tmp_det_sum += tf.reduce_mean(mag_f(tf.math.abs(pred_map[y_start:y_end,x_start,3] - tmp_tag))) # 3, right side
    if(use_cross and y_end-y_start>3):
      tmp_det_sum += tf.reduce_mean(mag_f(tf.math.abs(pred_map[(y_start+1):(y_end-1),x_start,0]))) # 0, up side
      tmp_det_sum += tf.reduce_mean(mag_f(tf.math.abs(pred_map[(y_start+1):(y_end-1),x_start,2]))) # 2, down side

    # Right boundary, same logic
    if((x_end-1)>x_start):
      tmp_det_sum += tf.reduce_mean(mag_f(tf.math.abs(pred_map[y_start:y_end,(x_end-1),1])))
      tmp_det_sum += tf.reduce_mean(mag_f(tf.math.abs(box[3] - float(x_end*stride_x) - pred_map[y_start:y_end,(x_end-1),3])))
      if(use_cross and y_end-y_start>3):
        tmp_det_sum += tf.reduce_mean(mag_f(tf.math.abs(pred_map[(y_start+1):(y_end-1),(x_end-1),0])))
        tmp_det_sum += tf.reduce_mean(mag_f(tf.math.abs(pred_map[(y_start+1):(y_end-1),(x_end-1),2])))

    abs_boundary_det+=[tmp_det_sum]

  return tf.convert_to_tensor(abs_boundary_det)

def pre_box_loss_by_msk(gt_mask, det_map, score_map, recf_size, det_map_fom='pet',
  lb_thr=0.2, use_cross=True, use_pixel=True, norl = True, mag_f='smooth', sigma=1.0):
  """
    Args:
      gt_mask: tensor ((1),hm,wm,(1)) mask.
      det_map: deta value tensor ((1),h,w,(num_class-1)*4)
      det_map_fom: det_map format, can be
        'pet': [dy1, dx1, dy2, dx2] percentage difference value in [-1.0,1.0]
        'pix': [dy1, dx1, dy2, dx2] pixel unit difference value
        'yxhw':[dy1, dx1, dh, dw] in float
      score_map: scores value tensor ((1),h,w,num_class)
      recf_size: receptive field size (rf_h,rf_w,stride_y,stride_x)
      lb_thr: lower boundary threshold in [0,0.5]
        calculate box with positive pixels higher than wh*wx*lb_thr
      use_pixel: 
        True: return pixel loss
        False: return box loss
      mag_f: (in box loss ONLY) magnification function, can be
        string: 'smooth', apply L1 smooth on loss function 
        string: 'sigmoid', apply sigmoid on loss function 
        tf.keras.layers.Lambda: apply input Lambda object
        others, don't apply any magnification function
      norl: True to use normalization difference
    Return:
      loss value
  """
  assert(recf_size.shape[0]>=4)
  num_class = int(det_map.shape[-1]/4)
  det_map_fom = det_map_fom.lower()
  if(not(det_map_fom in ['pet','pix','yxhw'])):det_map_fom='pet'
  pred_map = tf.reshape(det_map,det_map.shape[-3:])
  lb_thr = min(max(lb_thr,0.0),0.5)

  # convert gt_mask to [0,num_class-1]
  if(tf.reduce_max(gt_mask)>num_class):
    gt_mask = tf.cast(tf.cast(gt_mask,tf.bool),tf.int32)
  else:
    gt_mask = tf.cast(gt_mask,tf.int32)

  cube_h,cube_w,stride_y,stride_x = float(recf_size[0]),float(recf_size[1]),int(recf_size[2]),int(recf_size[3])
  icube_h,icube_w = int(cube_h), int(cube_w)
  min_pxls = int((icube_h*icube_w)*lb_thr)
  # min_pxls = min(int(icube_w*lb_thr),int(icube_h*lb_thr))

  # convert percentage difference to pixel difference
  if(det_map_fom=='pet'):pred_map = tf.stack([pred_map[:,:,0]*cube_h,pred_map[:,:,1]*cube_w,pred_map[:,:,2]*cube_h,pred_map[:,:,3]*cube_w],axis=-1)

  # select all boxes VS boundary boxes
  max_pxls = int(icube_h*icube_w) if use_pixel else int(icube_h*icube_w)-min_pxls
  tmp = len(gt_mask.shape)
  if(tmp==3):
    gt_mask_4d = tf.reshape(gt_mask,[1,]+gt_mask.shape)
    gt_mask = tf.reshape(gt_mask,gt_mask.shape[0:2])
  elif(tmp==2):
    gt_mask_4d = tf.reshape(gt_mask,[1,]+gt_mask.shape+[1])
  else:gt_mask_4d = gt_mask
  # gt_mask_4d = tf.image.resize(gt_mask_4d,[icube_h*pred_map.shape[-3],icube_w*pred_map.shape[-2]],'nearest')
  gt_mask_4d = tf.image.extract_patches(
    images=gt_mask_4d,
    sizes=[1,icube_h,icube_w,1],
    strides=[1,stride_y,stride_x,1],
    rates=[1,1,1,1],
    padding='SAME',
  )
  gt_mask_4d = gt_mask_4d[0,0:det_map.shape[-3],0:det_map.shape[-2],:]
  # 2D inc method
  gt_mask_4d = tf.reshape(gt_mask_4d, det_map.shape[-3:-1]+[icube_h,icube_w])
  gt_mask_4d_b = tf.cast(tf.cast(gt_mask_4d,tf.bool),tf.int32)
  ob_gt_mask = tf.reduce_sum(gt_mask_4d_b,axis=[-1,-2])
  boxs_main = tf.where(tf.math.logical_and(ob_gt_mask >= min_pxls, ob_gt_mask <= max_pxls))
  # label loss
  lmsk = tf.reduce_max(gt_mask_4d,axis=[-1,-2]).numpy()
  
  if(boxs_main.shape[0]==0 or boxs_main.shape[0]==None):
    # if can't find boxs_main, use max sub box
    # find max gtbox to avoid gradient disappearing
    tmp = tf.math.reduce_max(ob_gt_mask)
    boxs_main = tf.where(ob_gt_mask==tmp)
    lmsk[(ob_gt_mask!=tmp).numpy()]=0
  else:
    # if can find boxs_main, set ob_gt_mask<min_pxls as negtave
    # sometimes [ob_gt_mask<min_pxls] will got error
    lmsk[(ob_gt_mask<min_pxls).numpy()]=0

  if(score_map!=None):
    label_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=tf.reshape(score_map,[-1,score_map.shape[-1]]), 
      labels=tf.reshape(lmsk,[-1])))
  else:
    label_loss = None

  if(use_pixel):
    # pixel level difference
    pred_map_select = tf.gather_nd(pred_map,boxs_main)
    pred_mask = np.zeros(gt_mask.shape,dtype=np.int32)
    det_cx,det_cy = tf.meshgrid(
      tf.range(0,pred_map.shape[-2]+2,dtype=tf.float32)*cube_w,
      tf.range(0,pred_map.shape[-3]+2,dtype=tf.float32)*cube_h)

    for i in range(boxs_main.shape[0]):
      dy1,dy2 = pred_map_select[i,0::2]
      dx1,dx2 = pred_map_select[i,1::2]
      fy,fx = boxs_main[i]
      dx1,dx2 = int(dx1+det_cx[fy,fx]),int(dx2+det_cx[fy,fx])
      dy1,dy2 = int(dy1+det_cy[fy,fx]),int(dy2+det_cy[fy,fx])
      if(dy1<dy2 and dx1<dx2 and dy1>0 and dx1>0 and dy2<gt_mask.shape[0] and dx2<gt_mask.shape[1]):
        pred_mask[dy1:dy2,dx1:dx2] = 1
    gt_mask = gt_mask + pred_mask
    oara = tf.where(gt_mask==1)
    iara = tf.where(gt_mask==2)
    bxloss = tf.cast(tf.math.abs(iara.shape[0]-oara.shape[0]),tf.float32)
    if(norl):bxloss /= cube_h*cube_w
  else:
    # coordinate difference
    if(norl):
      # use normalization
      if(type(mag_f)==str):
        mag_f = mag_f.lower()
        if(mag_f=='tanh'):
          mag_f_y = tf.keras.layers.Lambda(lambda x:tf.math.sigmoid(x)/cube_h)
          mag_f_x = tf.keras.layers.Lambda(lambda x:tf.math.sigmoid(x)/cube_w)
        elif(mag_f=='smooth'):
          sigma_2 = sigma**2
          mag_f_y = tf.keras.layers.Lambda(lambda x:tf.where(x > (1. / x),x - (0.5 / sigma_2),tf.pow(x, 2) * (sigma_2 / 2.))/cube_h)
          mag_f_x = tf.keras.layers.Lambda(lambda x:tf.where(x > (1. / x),x - (0.5 / sigma_2),tf.pow(x, 2) * (sigma_2 / 2.))/cube_w)
        else:
          mag_f_y = tf.keras.layers.Lambda(lambda x:x/cube_h)
          mag_f_x = tf.keras.layers.Lambda(lambda x:x/cube_w)
      elif(mag_f == None):
        mag_f_y = tf.keras.layers.Lambda(lambda x:x/cube_h)
        mag_f_x = tf.keras.layers.Lambda(lambda x:x/cube_w)
      else:
        mag_f_y = tf.keras.layers.Lambda(lambda x:mag_f(x)/cube_h)
        mag_f_x = tf.keras.layers.Lambda(lambda x:mag_f(x)/cube_w)
    else:
      if(type(mag_f)==str):
        mag_f = mag_f.lower()
        if(mag_f=='tanh'):
          mag_f_y = tf.keras.layers.Lambda(lambda x: tf.math.sigmoid(x))
          mag_f_x = mag_f_y
        elif(mag_f=='smooth'):
          sigma_2 = sigma**2
          mag_f_y = tf.keras.layers.Lambda(lambda x: tf.where(x > (1. / x),x - (0.5 / sigma_2),tf.pow(x, 2) * (sigma_2 / 2.)))
          mag_f_x = mag_f_y
        else:
          mag_f_y = tf.keras.layers.Lambda(lambda x:x)
          mag_f_x = mag_f_y
      elif(mag_f == None):
        mag_f_y = tf.keras.layers.Lambda(lambda x:x)
        mag_f_x = mag_f_y
      else:
        mag_f_y = mag_f
        mag_f_x = mag_f
    bxloss = 0.0
    gt_mask_4d = tf.gather_nd(gt_mask_4d,boxs_main)
    pos_ins = tf.stop_gradient(tf.where(gt_mask_4d>0))
    ymin,xmin = int(icube_h*lb_thr),int(icube_w*lb_thr)
    ymax,xmax = icube_h-ymin,icube_w-xmin
    for i in range(boxs_main.shape[0]):
      fy,fx = boxs_main[i]
      my2,mx2 = tf.reduce_max(pos_ins[pos_ins[:,0]==i][:,1:3],axis=0)
      my1,mx1 = tf.reduce_min(pos_ins[pos_ins[:,0]==i][:,1:3],axis=0)
      dfy,dfx = 0,0
      if(my2<ymin):dfy=-1
      elif(my1>ymax):dfy=1
      if(mx2<xmin):dfx=-1
      elif(mx1>xmax):dfx=1

      if(dfy==0 and dfx==0):
        my2,my1 = float(icube_h-my2),float(my1)
        mx2,mx1 = float(icube_w-mx2),float(mx1)
        bxloss += mag_f_y(tf.math.abs(pred_map[fy,fx,0]-my1))
        bxloss += mag_f_y(tf.math.abs(pred_map[fy,fx,2]+my2)) # pdx2 should be negative
        bxloss += mag_f_x(tf.math.abs(pred_map[fy,fx,1]-mx1))
        bxloss += mag_f_x(tf.math.abs(pred_map[fy,fx,3]+mx2))
      elif(lmsk[fy,fx]>0):
        fy+=dfy
        fx+=dfx
        if(dfy>0):bxloss += mag_f_y(tf.math.abs(pred_map[fy,fx,0]+float(icube_h-my1)))
        elif(dfy<0):bxloss += mag_f_y(tf.math.abs(pred_map[fy,fx,2]-float(my2)))
        if(dfx>0):bxloss += mag_f_x(tf.math.abs(pred_map[fy,fx,1]+float(icube_w-mx1)))
        elif(dfx<0):bxloss += mag_f_x(tf.math.abs(pred_map[fy,fx,3]-float(mx2)))
    if(boxs_main.shape[0]>0):bxloss /= float(boxs_main.shape[0])
      
  return bxloss,label_loss

def gen_gt_from_msk(gt_mask,tar_size,recf_size,num_class,lb_thr=0.2):
  """
    Generate gt from original mask.
    Args:
      gt_mask: tensor ((1),hm,wm,(1)) mask.
      tar_size: target output map size [h,w]
      recf_size: receptive field size (rf_h,rf_w,stride_y,stride_x)
      lb_thr: lower boundary threshold in [0,0.5]
        calculate box with positive pixels higher than wh*wx*lb_thr
      num_class: number of class, include bg class
    Return:
      binary map in [0,numclass-1] with shape=dst_size, int32
  """
  assert(recf_size.shape[0]>=4)
  num_class-=1
  # convert gt_mask to [0,num_class-1]
  if(tf.reduce_max(gt_mask)>num_class):
    gt_mask = tf.cast(tf.cast(gt_mask,tf.bool),tf.int32)
  else:
    gt_mask = tf.cast(gt_mask,tf.int32)
  lb_thr = min(max(lb_thr,0.0),0.5)
  tmp = len(gt_mask.shape)
  if(tmp==3):gt_mask = tf.reshape(gt_mask,[1,]+gt_mask.shape)
  elif(tmp==2):gt_mask = tf.reshape(gt_mask,[1,]+gt_mask.shape+[1])
  else:gt_mask = gt_mask

  cube_h,cube_w,stride_y,stride_x = float(recf_size[0]),float(recf_size[1]),int(recf_size[2]),int(recf_size[3])
  icube_h,icube_w = int(cube_h), int(cube_w)
  min_pxls = int((icube_h*icube_w)*lb_thr)
  if(icube_h==1 and icube_w==1):return gt_mask

  gt_mask = tf.image.extract_patches(
    images=gt_mask,
    sizes=[1,icube_h,icube_w,1],
    strides=[1,stride_y,stride_x,1],
    rates=[1,1,1,1],
    padding='SAME',
  )
  gt_mask = gt_mask[0,0:tar_size[0],0:tar_size[1],:]
  gt_mask_b = tf.cast(tf.cast(gt_mask,tf.bool),tf.int32)
  gt_mask_b = tf.reduce_sum(gt_mask_b,axis=-1)
  gt_mask = tf.reduce_max(gt_mask,axis=-1).numpy()
  gt_mask[(gt_mask_b<min_pxls).numpy()]=0

  return gt_mask

@tf.function
def yxyx2area(boxes):
  """
  Return areas for boses
  """
  return (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
