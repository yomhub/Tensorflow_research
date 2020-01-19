import tensorflow as tf

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
def bbox_transform_inv_tf(boxes, deltas):
  boxes = tf.cast(boxes, deltas.dtype)
  widths = tf.subtract(boxes[:, 2], boxes[:, 0]) + 1.0
  heights = tf.subtract(boxes[:, 3], boxes[:, 1]) + 1.0
  ctr_x = tf.add(boxes[:, 0], widths * 0.5)
  ctr_y = tf.add(boxes[:, 1], heights * 0.5)

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