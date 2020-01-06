import os, sys
import tensorflow as tf
import numpy as np
import math

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

  return tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)