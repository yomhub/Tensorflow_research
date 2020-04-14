import os, sys
import tensorflow as tf
import numpy as np
import math
from tflib.log_tools import str2time, str2num, auto_scalar, auto_image
from tflib.evaluate_tools import draw_boxes, check_nan, draw_grid_in_gt, label_overlap_tf, gen_label_from_prob, draw_msk_in_gt
from tflib.bbox_transform import xywh2yxyx, gen_label_from_gt, gen_gt_from_msk
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
    with tf.GradientTape(persistent=False) as tape:
      tape.watch(self.model.trainable_variables)
      y_pred = self.model(x_single)
      loss_value = self.loss(y_single, y_pred)

    self.cur_loss += loss_value
    grads = tape.gradient(loss_value, self.model.trainable_variables)
    self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

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
    y_pred = self.model(x_single)

    y_pred = tf.cast(y_pred[:,:,:,1]>y_pred[:,:,:,0],tf.int32)
    y_pred = tf.reshape(y_pred,y_pred.shape+[1])
    tf.summary.image(
      name="Image in step {}.".format(self.current_step),
      data=x_single/256.0,step=0,max_outputs=1)
    tf.summary.image(
      name="Pred GT in step {}.".format(self.current_step),
      data=y_pred,step=0,max_outputs=1)

    y_single = tf.image.resize(y_single,y_pred.shape[-3:-1],'nearest')
    y_single=tf.cast(y_single[:,:,0]>0,tf.int32)
    y_single = tf.reshape(y_single,[1]+y_single.shape+[1])

    tf.summary.image(
      name="GT in step {}.".format(self.current_step),
      data=y_single,step=0,max_outputs=1)

    return 0

  def eval_callback(self,total_size,logger,time_usage):
    if(logger.writable()):
      logger.write("======================================\n")
      logger.write("Time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))
    return 0

    
