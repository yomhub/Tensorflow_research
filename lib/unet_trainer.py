import os, sys
import tensorflow as tf
import numpy as np
import math
from tflib.log_tools import str2time, str2num, auto_scalar, auto_image
from tflib.evaluate_tools import draw_boxes
from tflib.bbox_transform import xywh2yxyx, yxyx2xywh
from datetime import datetime
from trainer import Trainer

PROJ_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
LOGS_PATH = os.path.join(PROJ_PATH,"log")
MODEL_PATH = os.path.join(PROJ_PATH,"save_model")

class UnetTrainer(Trainer):
  def __init__(self,
    task_name,isdebug,
    logs_path = LOGS_PATH,
    model_path = MODEL_PATH,
    log_ft = True,
    log_grad = True):
    Trainer.__init__(self,task_name=task_name,isdebug=isdebug,logs_path = logs_path,model_path = model_path,)
    self.cur_loss = 0
    self.log_ft = bool(log_ft)
    self.log_grad = bool(log_grad)
    self.log_img_sec = ['ScoreMask','EdgeMask','GTMask','Image',]

  def train_action(self,x_single,y_single,step,logger):
    if(len(y_single['mask'].shape)==3):y_single['mask'] = tf.reshape(y_single['mask'],[1]+y_single['mask'].shape)
    if(y_single['mask'].dtype!=tf.float32 or y_single['mask'].dtype!=tf.float64):
      y_single['mask']=tf.cast(y_single['mask'],tf.float64)
    with tf.GradientTape(persistent=self.log_grad) as tape:
      tape.watch(self.model.trainable_variables)
      y_pred = self.model(x_single)
      loss_value = self.loss(y_single, y_pred)

    self.cur_loss += loss_value
    grads = tape.gradient(loss_value, self.model.trainable_variables)
    self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
    auto_scalar(loss_value,step,"Loss/Loss")
    auto_scalar(self.loss.cur_loss['mask'],step,"Loss/Mask loss")
    auto_scalar(self.loss.cur_loss['box'],step,"Loss/Box loss")
    auto_scalar(self.loss.cur_loss['score'],step,"Loss/Score loss")
    
    if(self.log_ft):
      auto_scalar(tf.reduce_mean(self.model.ft),step,"FT/ft mean")
      tmp = tf.reduce_mean(self.model.ft,axis=-1,keepdims=True)
      tmp = tf.math.l2_normalize(tmp,axis=(-1,-2,-3))
      tf.summary.image(
        name="Feature image",
        data=tmp,step=step,max_outputs=50)
    
    if(self.log_grad):
      for o in self.loss.cur_loss:
        tmp = tape.gradient(self.loss.cur_loss[o], self.model.trainable_variables)
        for i in range(len(tmp)):
          auto_scalar(
            tf.reduce_mean(tmp[i]) if(tmp[i]!=None)else 0.0,
            step,"Gradient {}/{}".format(o,self.model.trainable_variables[i].name))
    
    return 0

  def batch_callback(self,batch_size,logger,time_usage):
    self.cur_loss /= batch_size
    logger.write("======================================\n")
    logger.write("Step {}, size {}.\n".format(self.batch+1,batch_size))
    logger.write("Avg Loss: {}.\n".format(self.cur_loss))
    logger.write("Time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))
    logger.write("======================================\n\n")
    self.cur_loss = 0
    return 0

  def eval_action(self,x_single,y_single,step,logger):
    if(len(y_single['mask'].shape)==3):y_single['mask'] = tf.reshape(y_single['mask'],[1]+y_single['mask'].shape)
    if(y_single['mask'].dtype!=tf.float32 or y_single['mask'].dtype!=tf.float64):
      y_single['mask']=tf.cast(y_single['mask'],tf.float32)

    y_pred = self.model(x_single)
    mask = y_pred['mask']
    mask = tf.math.l2_normalize(mask,axis=(-1,-2,-3))
    mask = tf.broadcast_to(mask,mask.shape[:-1]+[3])
    y_mask = y_single['mask']
    y_mask = tf.image.resize(y_mask,mask.shape[-3:-1],'nearest')
    # add boundary enhance
    y_mask = tf.norm(tf.image.sobel_edges(y_mask),axis=-1)
    if(tf.reduce_max(y_mask)>1.0):y_mask/=256.0
    y_mask = tf.broadcast_to(y_mask,y_mask.shape[:-1]+[3])

    boxs = y_pred['gt'][y_pred['scr'][:,:,:,1]>y_pred['scr'][:,:,:,0]]
    boxs = tf.reshape(boxs,[-1,boxs.shape[-1]])

    if(tf.reduce_max(x_single)>1.0):x_single/=256.0
    x_single = tf.image.resize(x_single,mask.shape[-3:-1])
    if(boxs.shape[0]>0):x_single = draw_boxes(x_single,boxs,'cxywh')
    else:
      gt_boxs = y_pred['gt'][y_mask[:,:,:,0]>0.0]
      gt_boxs = tf.reshape(gt_boxs,[-1,gt_boxs.shape[-1]])
      x_single = draw_boxes(x_single,gt_boxs,'cxywh')

    scr = tf.where(y_pred['scr'][:,:,:,0]<y_pred['scr'][:,:,:,1], 1.0, 0.0)
    scr = tf.cast(scr,tf.float32)
    scr = tf.reshape(scr,scr.shape+[1])
    scr = tf.broadcast_to(scr,scr.shape[:-1]+[3])
    tmp = {
      'Image': x_single,
      'GTMask': y_mask,
      'EdgeMask': mask,
      'ScoreMask': scr,
    }
    tmp = tf.concat([tmp[o] for o in self.log_img_sec if o in tmp],axis=-2)

    tf.summary.image(
      name='|'.join(self.log_img_sec),
      # name="{} in step {}".format('|'.join(self.log_img_sec),step),
      data=tmp,step=step,max_outputs=50)

    # ft = tf.math.reduce_mean(self.model.ft,axis=-1,keepdims=True)
    # ft = tf.math.l2_normalize(ft,axis=-1)
    # if(tf.reduce_max(ft)>1.0):ft/=tf.reduce_max(ft)
    # tf.summary.image(
    #   name="FT",
    #   data=ft,step=step,max_outputs=50)
    
    return 0

  def eval_callback(self,total_size,logger,time_usage):
    if(logger.writable()):
      logger.write("======================================\n")
      logger.write("Time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))
    return 0

    
