import os, sys
import tensorflow as tf
import numpy as np
import math
from tflib.log_tools import str2time, str2num, auto_scalar, auto_image
from tflib.evaluate_tools import draw_boxes, check_nan, draw_grid_in_gt
from tflib.bbox_transform import xywh2yxyx, gen_label_from_gt
from tflib.img_tools import label_overlap_tf, gen_label_from_prob
from datetime import datetime
from trainer import Trainer

class LRCNNTrainer(Trainer):
  def __init__(self,task_name,isdebug,threshold=0.7,gtformat='yxyx'):
    Trainer.__init__(self,task_name=task_name,isdebug=isdebug)
    self.cur_loss = 0.0
    self.threshold = threshold
    if(gtformat.lower()=='yxyx' or gtformat.lower()=='yx'):
      self.gtformat='yxyx'
    else:
      self.gtformat='xywh'
    # super(FRCNNTrainer,self).__init__(kwargs)

  def train_action(self,x_single,y_single,step,logger):
    with tf.GradientTape(persistent=False) as tape:
      tape.watch(self.model.trainable_variables)
      y_pred = self.model(x_single)
      loss_value = self.loss(y_single, y_pred)

    self.cur_loss += loss_value
    grads = tape.gradient(loss_value, self.model.trainable_variables)
    self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    if(not(self.isdebug)):
      auto_scalar(loss_value,step,"Loss")
      for iname in self.loss.loss_detail:
        auto_scalar(self.loss.loss_detail[iname],step,"Loss detail: {}".format(iname))
    return 0
        
  def batch_callback(self,batch_size,logger,time_usage):
    self.cur_loss /= batch_size
    logger.write("======================================\n")
    logger.write("Batch {}, size {}.\n".format(self.batch+1,batch_size))
    logger.write("Avg Loss: {}.\n".format(self.cur_loss))
    logger.write("Time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))
    logger.write("======================================\n\n")
    self.cur_loss = 0
    return 0

  def draw_gt_pred_box(self,score,bbox,gt_box,image):
    imgh = float(image.shape[-3])
    imgw = float(image.shape[-2])
    col = tf.random.uniform(
      (1,3),
      minval=128,
      maxval=256,
    )

    l1prb = tf.reshape(score,[-1,2])
    l1prb = tf.nn.softmax(l1prb)
    l1box = tf.reshape(tf.gather(tf.reshape(bbox,[-1,4]),tf.where(l1prb>0)),[-1,4])
    # draw gt box frist
    ret = draw_grid_in_gt(score.shape[-3:-1],gt_box,image)
    if(l1box.shape[0]!=None and l1box.shape[0]>0):
      # if we have box, draw it
      l1box = tf.stack([l1box[:,0]/imgh,l1box[:,1]/imgw,l1box[:,2]/imgh,l1box[:,3]/imgw],axis=1)
      ret = tf.image.draw_bounding_boxes(ret,tf.reshape(l1box,[1,-1,4]),col)
    else:
      # or draw box in red forcibly
      l1prb = gen_label_from_gt(score.shape[1:3],gt_box,[imgh,imgw])
      l1prb = tf.reshape(l1prb,[-1])
      l1box = tf.reshape(tf.gather(tf.reshape(bbox,[-1,4]),tf.where(l1prb>0)),[-1,4])
      l1box = tf.stack([l1box[:,0]/imgh,l1box[:,1]/imgw,l1box[:,2]/imgh,l1box[:,3]/imgw],axis=1)
      ret = tf.image.draw_bounding_boxes(ret,tf.reshape(l1box,[1,-1,4]),tf.convert_to_tensor([[255.0,0.0,0.0]]))
    return ret

  def eval_action(self,x_single,y_single,step,logger):
    y_pred = self.model(x_single)
    # imgh = float(x_single.shape[-3])
    # imgw = float(x_single.shape[-2])
    if(self.gtformat=='xywh'):
      gt_box = xywh2yxyx(y_single[:,1:])
    else:
      gt_box = y_single[:,1:]
    label_gt = tf.stack([y_single[:,0],gt_box[:,0],gt_box[:,1],gt_box[:,2],gt_box[:,3]],axis=1)
    l1op = tf.reduce_sum(label_overlap_tf(gt_box,x_single.shape[1:3],gen_label_from_prob(y_pred["l1_score"])))
    l2op = tf.reduce_sum(label_overlap_tf(gt_box,x_single.shape[1:3],gen_label_from_prob(y_pred["l2_score"])))
    l3op = tf.reduce_sum(label_overlap_tf(gt_box,x_single.shape[1:3],gen_label_from_prob(y_pred["l3_score"])))
    auto_scalar(l1op,step,"L1 overlap")
    auto_scalar(l2op,step,"L2 overlap")
    auto_scalar(l3op,step,"L3 overlap")
    bx1_img = self.draw_gt_pred_box(y_pred["l1_score"],y_pred["l1_bbox"], label_gt, x_single)
    bx2_img = self.draw_gt_pred_box(y_pred["l2_score"],y_pred["l2_bbox"], label_gt, x_single)
    bx3_img = self.draw_gt_pred_box(y_pred["l3_score"],y_pred["l3_bbox"], label_gt, x_single)

    if(tf.reduce_max(bx1_img)>1.0):
      bx1_img = bx1_img/256.0
    if(tf.reduce_max(bx2_img)>1.0):
      bx2_img = bx2_img/256.0
    if(tf.reduce_max(bx3_img)>1.0):
      bx3_img = bx3_img/256.0
    tf.summary.image(name="Boxed image in L1 in step {}.".format(self.current_step),data=bx1_img,step=0,max_outputs=bx1_img.shape[0])
    tf.summary.image(name="Boxed image in L2 in step {}.".format(self.current_step),data=bx2_img,step=0,max_outputs=bx2_img.shape[0])
    tf.summary.image(name="Boxed image in L3 in step {}.".format(self.current_step),data=bx3_img,step=0,max_outputs=bx3_img.shape[0])
    return 0

  def eval_callback(self,total_size,logger,time_usage):
    if(logger.writable()):
      logger.write("======================================\n")
      logger.write("Time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))
    return 0

    
