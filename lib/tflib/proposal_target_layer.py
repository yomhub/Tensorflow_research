import tensorflow as tf
import numpy as np
from tflib.bbox_transform import bbox_transform
from tflib.overlap import overlap_tf

def proposal_target_layer_tf(rpn_rois, rpn_scores, gt_boxes, _num_classes, settings):
  """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  Args:

    gt_boxes: Tensor with shape (total_gts,5)
      where 4 is [xstart, ystart, w, h, class]
  """
  # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
  # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
  all_rois = rpn_rois
  all_scores = rpn_scores

  # Include ground-truth boxes in the set of candidate rois
  if settings["USE_GT"]:
    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    all_rois = np.vstack(
      (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
    )
    # not sure if it a wise appending, but anyway i am not using it
    all_scores = np.vstack((all_scores, zeros))

  num_images = 1
  rois_per_image = settings["BATCH_SIZE"] / num_images
  fg_rois_per_image = np.round(settings["FG_FRACTION"] * rois_per_image)

  # Sample rois with classification labels and bounding box regression
  # targets
  labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
    all_rois, all_scores, gt_boxes, fg_rois_per_image,
    rois_per_image, _num_classes)

  rois = rois.reshape(-1, 5)
  roi_scores = roi_scores.reshape(-1)
  labels = labels.reshape(-1, 1)
  bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
  bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
  bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

  return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
  """Generate a random sample of RoIs comprising foreground and background
  examples.
  """
  # overlaps: (rois x gt_boxes)
  overlaps = overlap_tf(all_rois,gt_boxes)
  # find maximum in overlaps above gt_boxes
  gt_assignment = tf.argmax(overlaps,axis=1)
  max_overlaps = tf.reduce_max(overlaps,axis=1)
  labels = tf.gather(gt_boxes,gt_assignment)

  # Select foreground RoIs as those with >= FG_THRESH overlap
  fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
  # Guard against the case when an image has fewer than fg_rois_per_image
  # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
  bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                     (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]

  # Small modification to the original version where we ensure a fixed number of regions are sampled
  if fg_inds.size > 0 and bg_inds.size > 0:
    fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
    fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
    bg_rois_per_image = rois_per_image - fg_rois_per_image
    to_replace = bg_inds.size < bg_rois_per_image
    bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
  elif fg_inds.size > 0:
    to_replace = fg_inds.size < rois_per_image
    fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
    fg_rois_per_image = rois_per_image
  elif bg_inds.size > 0:
    to_replace = bg_inds.size < rois_per_image
    bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
    fg_rois_per_image = 0
  else:
    import pdb
    pdb.set_trace()

  # The indices that we're selecting (both fg and bg)
  keep_inds = np.append(fg_inds, bg_inds)
  # Select sampled values from various arrays:
  labels = labels[keep_inds]
  # Clamp labels for the background RoIs to 0
  labels[int(fg_rois_per_image):] = 0
  rois = all_rois[keep_inds]
  roi_scores = all_scores[keep_inds]

  bbox_target_data = _compute_targets(
    rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

  bbox_targets, bbox_inside_weights = \
    _get_bbox_regression_labels(bbox_target_data, num_classes)

  return labels, rois, roi_scores, bbox_targets, bbox_inside_weights