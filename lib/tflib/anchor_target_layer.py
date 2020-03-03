import tensorflow as tf
import numpy as np
from tflib.bbox_transform import bbox_transform, clip_boxes_tf, map2coordinate
from tflib.evaluate_tools import overlap_tf, check_inside

def anchor_target_layer_tf(all_anchors, gt_boxes, im_info, settings):
  """
    TF function version anchor target layer
    Args:
      all_anchors: Tensor with shape (1,h,w,num_anchor_pre_point*4)
        where 4 is [y1, x1, y2, x2]
      gt_boxes: Tensor with shape (total_gts,4)
        where 4 is [y1,x1,y2,x2]
      im_info: [height，width]
      settings:{
        "RPN_NEGATIVE_OVERLAP" : RPN_NEGATIVE_OVERLAP,
        "RPN_POSITIVE_OVERLAP" : RPN_POSITIVE_OVERLAP,
          negative / positive overlap threshold
        "RPN_CLOBBER_POSITIVES" : RPN_CLOBBER_POSITIVES,
          If an anchor satisfied by positive and 
          negative conditions set to negative
        "RPN_BATCHSIZE" : RPN_BATCHSIZE,
          Total number of examples
        "RPN_FG_FRACTION" : RPN_FG_FRACTION,
          Max percentage of foreground examples
        "RPN_BBOX_INSIDE_WEIGHTS" : RPN_BBOX_INSIDE_WEIGHTS,
          Deprecated (outside weights)
        "RPN_POSITIVE_WEIGHT", RPN_POSITIVE_WEIGHT
          Give the positive RPN examples weight 
          of p * 1 / {num positives}
          and give negatives a weight of (1 - p)
          Set to -1.0 to use uniform example weighting
        ...
      }
    Return:
      Tensors
      rpn_labels with shape (total_anchers, 1)
      where True is 1, gt is 0, bg is -1
      rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
      with shape (total_anchers, 4)
  """
  all_anchors = tf.reshape(all_anchors,[-1,4])
  # inside_len = inds_inside.shape[0]
  inds_inside = tf.where(
    (all_anchors[:,0]>=0.0) &
    (all_anchors[:,1]>=0.0) &
    (all_anchors[:,2]<=im_info[0]) &
    (all_anchors[:,3]<=im_info[1])
  )
  inds_inside = tf.reshape(inds_inside,[-1])
  boxs_inside = tf.gather(all_anchors,inds_inside)

  # boxs_inside shape (inside_len,4)
  # gt_boxs shape (total_gts,4)
  # overlaps shape (inside_len,total_gts)
  overlaps = overlap_tf(boxs_inside, gt_boxes)

  # just slect the max value
  # of overlaps in axis = (0,1)
  # argmax_overlaps: shape=(inside_len,1) value=[0,gt_box_len)
  # max_overlaps: (inside_len,1) max overlaps in each boxs_inside
  # gt_argmax_overlaps: shape=(gt_box_len,1) value=[0,inside_len)
  # gt_max_overlaps: (gt_box_len,1) max overlaps in each gt_boxs

  argmax_overlaps = tf.argmax(overlaps,axis=1,output_type=tf.int32)
  gt_argmax_overlaps = tf.argmax(overlaps,axis=0,output_type=tf.int32)
  max_overlaps = tf.reduce_max(overlaps,axis=1)
  # gt_max_overlaps = tf.reduce_max(overlaps,axis=0)

  # since tf.assign will unavailable in tf2+
  # we use numpy instead of tensor
  # labels: (inside_len,1) labels for each boxs_inside
  labels = np.full([inds_inside.shape[0]],-1)
  if not settings["RPN_CLOBBER_POSITIVES"]:
    # assign bg labels first so that positive labels can clobber them
    # first set the negatives
    labels[max_overlaps.numpy() < settings["RPN_NEGATIVE_OVERLAP"]] = 0

  # fg label: for each gt, anchor with highest overlap
  labels[gt_argmax_overlaps.numpy()] = 1

  # fg label: above threshold IOU
  labels[max_overlaps.numpy()>settings["RPN_POSITIVE_OVERLAP"]] = 1
  
  if settings["RPN_CLOBBER_POSITIVES"]:
    # assign bg labels last so that negative labels can clobber positives
    labels[max_overlaps.numpy() < settings["RPN_NEGATIVE_OVERLAP"]] = 0

  # subsample positive labels if we have too many
  num_fg = int(settings["RPN_FG_FRACTION"] * settings["RPN_BATCHSIZE"])
  fg_inds = np.where(labels == 1)[0]
  if len(fg_inds) > num_fg:
    disable_inds = np.random.choice(
      fg_inds, size=(len(fg_inds) - num_fg), replace=False)
    labels[disable_inds] = -1

  # subsample negative labels if we have too many
  num_bg = settings["RPN_BATCHSIZE"] - np.sum(labels == 1)
  bg_inds = np.where(labels == 0)[0]
  if len(bg_inds) > num_bg:
    disable_inds = np.random.choice(
      bg_inds, size=(len(bg_inds) - num_bg), replace=False)
    labels[disable_inds] = -1  

  # bbox_targets = bbox_transform(boxs_inside, tf.gather(gt_boxes,argmax_overlaps))
  bbox_targets=boxs_inside
  # only the positive ones have regression targets
  bbox_inside_weights = np.zeros((overlaps.shape[0], 4), dtype=np.float32)
  bbox_inside_weights[labels == 1, :] = np.array(settings["RPN_BBOX_INSIDE_WEIGHTS"])

  bbox_outside_weights = np.zeros((overlaps.shape[0], 4), dtype=np.float32)
  if settings["RPN_POSITIVE_WEIGHT"] < 0:
    # uniform weighting of examples (given non-uniform sampling)
    num_examples = np.sum(labels >= 0)
    positive_weights = np.ones((1, 4)) * 1.0 / num_examples
    negative_weights = np.ones((1, 4)) * 1.0 / num_examples
  else:
    assert ((settings["RPN_POSITIVE_WEIGHT"] > 0) &
            (settings["RPN_POSITIVE_WEIGHT"] < 1))
    positive_weights = (settings["RPN_POSITIVE_WEIGHT"] /
                        np.sum(labels == 1))
    negative_weights = ((1.0 - settings["RPN_POSITIVE_WEIGHT"]) /
                        np.sum(labels == 0))
  bbox_outside_weights[labels == 1, :] = positive_weights
  bbox_outside_weights[labels == 0, :] = negative_weights
  
  # map up to original set of anchors
  total_anchors = all_anchors.shape[0]
  labels = _unmap(labels, total_anchors, inds_inside.numpy(), fill=-1)
  bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside.numpy(), fill=0)
  bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside.numpy(), fill=0)
  # bbox_inside_weights = bbox_inside_weights.reshape([1,]+anchor_img_size+[anchor_size*4])
  bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside.numpy(), fill=0)
  # bbox_outside_weights = bbox_outside_weights.reshape([1,]+anchor_img_size+[anchor_size*4])
  # convert all numpy to tensor
  rpn_labels = tf.convert_to_tensor(labels,dtype=tf.int32)
  rpn_bbox_targets = tf.convert_to_tensor(bbox_targets,dtype=tf.float32)
  rpn_bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights,dtype=tf.float32)
  rpn_bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights,dtype=tf.float32)

  return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

def point_anchor_target_layer_tf(all_anchors, gt_boxes, im_info, settings):
  """
    Args:
      all_anchors: Tensor with shape (1,h,w,num_anchor_pre_point*4)
        where 4 is [y1, x1, y2, x2]
      gt_boxes: Tensor with shape (total_gts,4)
        where 4 is [y1, x1, y2, x2]
      all_score: Tensor with shape (1,h,w,num_anchor_pre_point*2)
        where 2 is [negitive, positive,...]
      im_info: [height，width]
  """
  feat_h = all_anchors.shape[1]
  feat_w = all_anchors.shape[2]
  anc_num = int(all_anchors.shape[3]/4)

  # map gtbox to feature coordinate
  maped_anc = map2coordinate(gt_boxes,im_info,all_anchors.shape[1:3])
  # convert [1,w]->[0,w-1]
  maped_anc_int = tf.cast(maped_anc,tf.int32).numpy()-1

  mask_np = np.full((feat_h,feat_w,anc_num),-1,dtype=np.int32)

  # positive labels pre gt_box
  num_fg = int(settings["RPN_FG_FRACTION"] * settings["RPN_BATCHSIZE"] / gt_boxes.shape[0])

  cont = 0
  for box in maped_anc_int:
    if(box[2]>=box[0] and box[3]>=box[1]):
      sub_h = box[2]-box[0]+1
      sub_w = box[3]-box[1]+1
      x_start,x_end = box[1],box[3]
      y_start,y_end = box[0],box[2]
      inside = tf.reshape(all_anchors[0,box[0]:box[2]+1,box[1]:box[3]+1,:],[-1,4])
      in_mask = check_inside(inside,im_info)
      inside = tf.where(
          tf.tile(tf.reshape(in_mask,[in_mask.shape[0],1]),[1,4]),
          inside,tf.zeros_like(inside)
        )
    else:
      continue

    sub_mask = np.full((sub_h*sub_w*anc_num),-1,dtype=np.int8)
    # ovelap [insideY*insideX*num_anc,1]
    ovelap = overlap_tf(inside,tf.reshape(maped_anc[cont],[1,4]))
    ovelap = tf.reshape(ovelap,[-1])
    if not settings["RPN_CLOBBER_POSITIVES"]:
      # assign bg labels first so that positive labels can clobber them
      # first set the negatives
      sub_mask[ovelap.numpy() < settings["RPN_NEGATIVE_OVERLAP"]] = 0

    values, indices_k = tf.math.top_k(ovelap,k=num_fg)
    indices_k = tf.gather(indices_k,tf.where(values>settings["RPN_POSITIVE_OVERLAP"]))
    sub_mask[indices_k.numpy()] = 1

    gt_argmax = tf.argmax(ovelap,output_type=tf.int32)
    sub_mask[gt_argmax.numpy()] = 1

    if settings["RPN_CLOBBER_POSITIVES"]:
      # assign bg labels last so that negative labels can clobber positives
      sub_mask[ovelap.numpy() < settings["RPN_NEGATIVE_OVERLAP"]] = 0

    sub_mask = np.pad(
      sub_mask.reshape((sub_h,sub_w,anc_num)),
      # in axis (sub_h,sub_w,anc_num)
      pad_width=((y_start,feat_h-y_end-1),(x_start,feat_w-x_end-1),(0,0)),
      mode='constant',
      )
    
    mask_np[sub_mask==1]=1
    mask_np[sub_mask==0]=0
    cont += 1
  
  mask_np = mask_np.reshape([-1])

  # subsample negative labels if we have too many
  num_bg = settings["RPN_BATCHSIZE"] - np.sum(mask_np == 1)
  bg_inds = np.where(mask_np == 0)[0]
  if len(bg_inds) > num_bg:
    disable_inds = np.random.choice(
      bg_inds, size=(len(bg_inds) - num_bg), replace=False)
    mask_np[disable_inds] = -1 

  labels = tf.convert_to_tensor(mask_np,dtype=tf.int32)
  inc = tf.reshape(tf.where(labels!=-1),[-1])
  bbox_targets = tf.gather(tf.reshape(all_anchors,[-1,4]),inc)
  mask_np = mask_np[inc.numpy()]
  
  # only the positive ones have regression targets
  bbox_inside_weights = np.zeros((bbox_targets.shape[0], 4), dtype=np.float32)
  bbox_inside_weights[mask_np == 1, :] = np.array(settings["RPN_BBOX_INSIDE_WEIGHTS"])

  bbox_outside_weights = np.zeros((bbox_targets.shape[0], 4), dtype=np.float32)
  if settings["RPN_POSITIVE_WEIGHT"] < 0:
    # uniform weighting of examples (given non-uniform sampling)
    num_examples = np.sum(mask_np >= 0)
    positive_weights = np.ones((1, 4)) * 1.0 / num_examples
    negative_weights = np.ones((1, 4)) * 1.0 / num_examples
  else:
    assert ((settings["RPN_POSITIVE_WEIGHT"] > 0) &
            (settings["RPN_POSITIVE_WEIGHT"] < 1))
    positive_weights = (settings["RPN_POSITIVE_WEIGHT"] /
                        np.sum(mask_np == 1))
    negative_weights = ((1.0 - settings["RPN_POSITIVE_WEIGHT"]) /
                        np.sum(mask_np == 0))

  bbox_outside_weights[mask_np == 1, :] = positive_weights
  bbox_outside_weights[mask_np == 0, :] = negative_weights
  bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights,dtype=tf.float32)
  bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights,dtype=tf.float32)

  return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def _unmap(data, count, inds, fill=0):
  """ 
    Unmap a subset of item (data) back to the original set of items (of
    size count)
    Args: 
      data: ndarray of data
      count: size of frist aix
      inds: index of data in output array
      fill: fill data
    Return:
      ndarray with (count,data.shape)
  """
  if len(data.shape) == 1:
    ret = np.full((count,), fill, dtype=np.float32)
    ret[inds] = data
  else:
    ret = np.full([count,] + list(data.shape[1:]), fill, dtype=np.float32)
    ret[inds, :] = data
  return ret