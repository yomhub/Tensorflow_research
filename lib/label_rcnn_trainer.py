import os, sys
import tensorflow as tf
import numpy as np
from tflib.log_tools import _str2time, _str2num, _auto_scalar, _auto_image
from tflib.evaluate_tools import draw_boxes, check_nan
from datetime import datetime
from trainer import Trainer

class LRCNNTrainer(Trainer):
  def __init__(self,task_name,isdebug,threshold=0.8):
    Trainer.__init__(self,task_name=task_name,isdebug=isdebug)
    self.cur_loss = 0.0
    self.threshold = threshold
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
      _auto_scalar(loss_value,step,"Loss")
      for iname in self.loss.loss_detail:
        _auto_scalar(self.loss.loss_detail[iname],step,"Loss detail: {}".format(iname))
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

  def draw_gt_pred_box(self,y_pred,gt_box,image):
    gt_box = tf.reshape(gt_box,[1,]+gt_box.shape[-2:])
    imgh = float(image.shape[-3])
    imgw = float(image.shape[-2])
    ret = tf.image.draw_bounding_boxes(image,gt_box,tf.convert_to_tensor([[1.0,0.0,0.0]]))
    col = tf.random.uniform(
      (3,1,3),
      minval=0,
      maxval=254,
    )
    l1prb = tf.reshape(y_pred["l1_score"],[-1,2])
    l1box = tf.reshape(tf.gather(tf.reshape(y_pred["l1_bbox"],[-1,4]),tf.where(l1prb[:,1]>self.threshold)),[-1,4])
    l1box = tf.stack([l1box[:,0]/imgh,l1box[:,1]/imgw,l1box[:,2]/imgh,l1box[:,3]/imgw],axis=1)
    l2prb = tf.reshape(y_pred["l2_score"],[-1,2])
    l2box = tf.reshape(tf.gather(tf.reshape(y_pred["l2_bbox"],[-1,4]),tf.where(l2prb[:,1]>self.threshold)),[-1,4])
    l2box = tf.stack([l2box[:,0]/imgh,l2box[:,1]/imgw,l2box[:,2]/imgh,l2box[:,3]/imgw],axis=1)
    l3prb = tf.reshape(y_pred["l3_score"],[-1,2])
    l3box = tf.reshape(tf.gather(tf.reshape(y_pred["l3_bbox"],[-1,4]),tf.where(l3prb[:,1]>self.threshold)),[-1,4])
    l3box = tf.stack([l3box[:,0]/imgh,l3box[:,1]/imgw,l3box[:,2]/imgh,l3box[:,3]/imgw],axis=1)

    if(l1box.shape[0]!=None and l1box.shape[0]>0):
      ret = tf.image.draw_bounding_boxes(ret,tf.reshape(l1box,[1,-1,4]),col[0])
    if(l2box.shape[0]!=None and l2box.shape[0]>0):
      ret = tf.image.draw_bounding_boxes(ret,tf.reshape(l2box,[1,-1,4]),col[1])
    if(l3box.shape[0]!=None and l3box.shape[0]>0):
      ret = tf.image.draw_bounding_boxes(ret,tf.reshape(l3box,[1,-1,4]),col[2])
    return ret

  def eval_action(self,x_single,y_single,step,logger):
    y_pred = self.model(x_single)
    bx_img = self.draw_gt_pred_box(y_pred, y_single[:,1:], x_single)
    with self.file_writer.as_default():
      tf.summary.image(name="Boxed image in step {}.".format(step),data=bx_img/256.0,step=0)
    return 0

  def eval_callback(self,total_size,logger,time_usage):
    if(logger.writable()):
      logger.write("======================================\n")
      logger.write("Time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))
    return 0

    
