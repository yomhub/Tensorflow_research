import os, sys
import tensorflow as tf
import numpy as np
from tflib.log_tools import _str2time, _str2num, _auto_scalar, _auto_image
from tflib.evaluate_tools import draw_boxes, check_nan
from datetime import datetime
from trainer import Trainer

class FRCNNTrainer(Trainer):
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

    gt_prob = y_pred["rpn_cls_score1"]
    gt_prob = tf.reshape(gt_prob,[-1,gt_prob.shape[-1]])
    gt_prob = gt_prob[:,int(gt_prob.shape[-1]/2):]
    gt_prob = tf.reshape(gt_prob,[-1])
    gt_boxes = y_pred["rpn_bbox_pred1"]
    gt_boxes = tf.reshape(gt_boxes,[-1,4])
    gt_in = tf.reshape(tf.where(gt_prob>0.8),[-1])
    bbx = tf.gather(gt_boxes,gt_in)
    gtbox_num = int(gt_in.shape[0])

    if(not(self.isdebug)):
      _auto_scalar(loss_value,step,"Loss")
      _auto_scalar(self.loss.loss_detail,step)
      _auto_scalar(gtbox_num, step, "GT_box_num_pred")
      _auto_scalar(y_single.shape[0], step, "GT_box_num_true")
      _auto_scalar(gtbox_num / int(y_single.shape[0]), step, "GT_box_num_pred_div_true")
      if(gtbox_num>0 and int(y_single.shape[0]*2)>gtbox_num):
        bximg = draw_boxes(x_single,bbx,'yxyx')
        tf.summary.image("boxed_images in step {}".format(step),tf.cast(bximg,tf.int32),step)

    self.cur_rpn_cross_entropy += self.loss.loss_detail["rpn_cross_entropy"]
    self.cur_rpn_loss_box += self.loss.loss_detail["rpn_loss_box"]
    self.cur_cross_entropy += self.loss.loss_detail["cross_entropy"]
    self.cur_loss_box += self.loss.loss_detail["loss_box"]
    self.cur_loss += loss_value
    self.cur_gtbox_num += gtbox_num

    grads = tape.gradient(loss_value, self.model.trainable_variables)

    if(True):
      had_nan = False
      for iname in self.loss.loss_detail:
        self.grad_dict[iname]=tape.gradient(self.loss.loss_detail[iname], self.model.trainable_variables)
        nan_ind = check_nan(self.grad_dict[iname])
        if(type(nan_ind)==list or nan_ind!=0):
          logger.write("======================================\n")
          logger.write("Get NAN at batch {} setp {}.\n".format(self.batch+1,step))
          logger.write("From loss: {}, loss value: {}.\n".format(iname,self.loss.loss_detail[iname]))
          for iid in nan_ind:
            logger.write("\tGradient by {} has {} Nan.\n".format(self.model.trainable_variables[iid[0]].name,iid[1]))
          if(iname!="loss_box" and iname!="cross_entropy"):
            had_nan = True
      if(had_nan):
        logger.close()
        return -1
    
    if(self.isdebug):  
      self.opt.apply_gradients(zip(self.grad_dict["rpn_cross_entropy"], self.model.trainable_variables))
      self.opt.apply_gradients(zip(self.grad_dict["rpn_loss_box"], self.model.trainable_variables))
      self.opt.apply_gradients(zip(self.grad_dict["rpn_cross_entropy1"], self.model.trainable_variables))
      self.opt.apply_gradients(zip(self.grad_dict["rpn_loss_box1"], self.model.trainable_variables))
      # self.opt.apply_gradients(zip(grads, model.trainable_variables))
    else:
      self.opt.apply_gradients(zip(self.grad_dict["rpn_cross_entropy"], self.model.trainable_variables))
      self.opt.apply_gradients(zip(self.grad_dict["rpn_loss_box"], self.model.trainable_variables))
      self.opt.apply_gradients(zip(self.grad_dict["rpn_cross_entropy1"], self.model.trainable_variables))
      self.opt.apply_gradients(zip(self.grad_dict["rpn_loss_box1"], self.model.trainable_variables))

    return 0

  def batch_callback(self,batch_size,logger,time_usage):
    self.cur_rpn_cross_entropy /= batch_size
    self.cur_rpn_loss_box /= batch_size
    self.cur_cross_entropy /= batch_size
    self.cur_loss_box /= batch_size
    self.cur_loss /= batch_size
    self.cur_gtbox_num /= batch_size

    logger.write("======================================\n")
    logger.write("Batch {}, size {}.\n".format(self.batch+1,batch_size))
    logger.write("Avg Loss: {}.\n".format(self.cur_loss))
    logger.write("Avg cur_rpn_cross_entropy = {}.\n".format(self.cur_rpn_cross_entropy))
    logger.write("Avg cur_rpn_loss_box = {}.\n".format(self.cur_rpn_loss_box))
    logger.write("Avg cur_cross_entropy = {}.\n".format(self.cur_cross_entropy))
    logger.write("Avg cur_loss_box = {}.\n".format(self.cur_loss_box))
    logger.write("Avg cur_gtbox_num = {}.\n".format(self.cur_gtbox_num))
    logger.write("Time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))
    logger.write("======================================\n\n")

    self.cur_rpn_cross_entropy = 0.0
    self.cur_rpn_loss_box = 0.0
    self.cur_cross_entropy = 0.0
    self.cur_loss_box = 0.0
    self.cur_loss = 0.0
    self.cur_gtbox_num = 0.0
