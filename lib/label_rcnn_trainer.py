import os, sys
import tensorflow as tf
import numpy as np
from tflib.log_tools import _str2time, _str2num, _auto_scalar, _auto_image
from tflib.evaluate_tools import draw_boxes, check_nan
from datetime import datetime
from trainer import Trainer

class LabelRCNNTrainer(Trainer):
  def __init__(self,task_name,isdebug):
    Trainer.__init__(self,task_name=task_name,isdebug=isdebug)
    self.cur_rpn_cross_entropy = 0.0
    self.cur_rpn_loss_box = 0.0
    self.cur_cross_entropy = 0.0
    self.cur_loss_box = 0.0
    self.cur_loss = 0.0
    self.cur_gtbox_num = 0.0
    # super(FRCNNTrainer,self).__init__(kwargs)

  def train_action(self,x_single,y_single,step,logger):
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(self.model.trainable_variables)
      y_pred = self.model(x_single)
      loss_value = self.loss(y_single, y_pred)
    
    grads = tape.gradient(loss_value, self.model.trainable_variables)
    self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    if(not(self.isdebug)):
      _auto_scalar(loss_value,step,"Loss")
