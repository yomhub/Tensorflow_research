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
    model_path = MODEL_PATH,):
    Trainer.__init__(self,task_name=task_name,isdebug=isdebug,logs_path = logs_path,model_path = model_path,)
    self.cur_loss = 0

  def train_action(self,x_single,y_single,step,logger):
    if(len(y_single['mask'].shape)==3):y_single['mask'] = tf.reshape(y_single['mask'],[1]+y_single['mask'].shape)
    if(y_single['mask'].dtype!=tf.float32 or y_single['mask'].dtype!=tf.float64):
      y_single['mask']=tf.cast(y_single['mask'],tf.float64)
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(self.model.trainable_variables)
      y_pred = self.model(x_single)
      loss_value = self.loss(y_single, y_pred)

    self.cur_loss += loss_value
    grads = tape.gradient(loss_value, self.model.trainable_variables)
    self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
    auto_scalar(loss_value,step,"Loss")
    auto_scalar(self.loss.cur_loss['mask'],step,"Mask loss")
    auto_scalar(self.loss.cur_loss['box'],step,"Box loss")
    auto_scalar(self.loss.cur_loss['score'],step,"Score loss")
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

  def eval_action(self,x_single,y_single,step,logger):
    if(len(y_single['mask'].shape)==3):y_single['mask'] = tf.reshape(y_single['mask'],[1]+y_single['mask'].shape)
    if(y_single['mask'].dtype!=tf.float32 or y_single['mask'].dtype!=tf.float64):
      y_single['mask']=tf.cast(y_single['mask'],tf.float32)

    y_pred = self.model(x_single)
    mask = y_pred['mask']
    if(tf.reduce_max(mask)>1.0):mask/=256.0
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
    tmp = tf.concat([x_single,
      # tf.broadcast_to(y_single,y_single.shape[:-1]+[3]),
      y_mask,
      # tf.broadcast_to(y_pred,y_pred.shape[:-1]+[3])],
      mask
      ],
      axis=-2)

    tf.summary.image(
      name="Image|GT|Pred in step {}.".format(self.current_step),
      data=tmp,step=0,max_outputs=1)

    
    return 0

  def eval_callback(self,total_size,logger,time_usage):
    if(logger.writable()):
      logger.write("======================================\n")
      logger.write("Time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))
    return 0

    
