import os, sys
import tensorflow as tf
import numpy as np
from model.config import cfg
from model.faster_rcnn import Faster_RCNN, RCNNLoss
from datetime import datetime

logs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"log")

class Trainer():
  def __init__(self,  
    logs_path = logs_path,
    task_name = None,
  ):
    """
    Args:
      logs_path: Tensor board dir.
      task_name: Floder name under logs_path.

    """
    super(Trainer,self).__init__()
    self.logs_path = logs_path if task_name == None else os.path.join(logs_path,task_name)
    self.file_writer = tf.summary.create_file_writer(
      os.path.join(self.logs_path,datetime.now().strftime("%Y%m%d-%H%M%S")))
    self.file_writer.set_as_default()
    self.current_step = 0
    self.batch = 0
    
  def log_image(self, tfimg, log_num=10, img_size=None):
    with self.file_writer.as_default():
    # Don't forget to reshape.
      if(img_size!=None):
        tfimg = tf.reshape(tfimg, (-1, img_size[0], img_size[1], tfimg.shape[-1]))
      tf.summary.image("25 training data examples", tfimg, max_outputs=log_num, step=0)
  
  def fit(self,
    x_train,y_train,
    model,loss,opt,
    x_val=None,y_val=None,
  ):
    tstart = datetime.now()
    total_data = x_train.shape[0]
    cur_stp = self.current_step
    if(type(y_train)==list):
      assert(total_data==len(y_train))
      
    if(total_data>1):
      x_train = tf.split(x_train,total_data,axis=0)
      cur_rpn_cross_entropy = 0.0
      cur_rpn_loss_box = 0.0
      cur_cross_entropy = 0.0
      cur_loss_box = 0.0
      cur_loss = 0.0
      for step in range(total_data):
        with tf.GradientTape(persistent=True) as tape:
          tape.watch(model.trainable_variables)
          y_pred = model(x_train[step])
          loss_value = loss(y_train[step], y_pred)

        tf.summary.scalar("Loss",loss_value,step=cur_stp + step)
        tf.summary.scalar("rpn_cross_entropy loss",loss.loss_detail["rpn_cross_entropy"],step=cur_stp + step)
        tf.summary.scalar("rpn_loss_box loss",loss.loss_detail["rpn_loss_box"],step=cur_stp + step)
        tf.summary.scalar("cross_entropy loss",loss.loss_detail["cross_entropy"],step=cur_stp + step)
        tf.summary.scalar("loss_box loss",loss.loss_detail["loss_box"],step=cur_stp + step)

        cur_rpn_cross_entropy += loss.loss_detail["rpn_cross_entropy"]
        cur_rpn_loss_box += loss.loss_detail["rpn_loss_box"]
        cur_cross_entropy += loss.loss_detail["cross_entropy"]
        cur_loss_box += loss.loss_detail["loss_box"]
        cur_loss += loss_value

        grads = tape.gradient(loss_value, model.trainable_variables)
        g_rpn_cross_entropy = tape.gradient(loss.loss_detail["rpn_cross_entropy"], model.trainable_variables)
        g_rpn_loss_box = tape.gradient(loss.loss_detail["rpn_loss_box"], model.trainable_variables)
        g_cross_entropy = tape.gradient(loss.loss_detail["cross_entropy"], model.trainable_variables)
        g_loss_box = tape.gradient(loss.loss_detail["loss_box"], model.trainable_variables)

        opt.apply_gradients(zip(g_rpn_cross_entropy, model.trainable_variables))
        opt.apply_gradients(zip(g_rpn_loss_box, model.trainable_variables))

      cur_stp += step
      cur_rpn_cross_entropy /= total_data
      cur_rpn_loss_box /= total_data
      cur_cross_entropy /= total_data
      cur_loss_box /= total_data
      cur_loss /= total_data

    else:
      cur_stp += 1
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        y_pred = model(x_train[step])
        loss_value = loss(y_train[step], y_pred)

      tf.summary.scalar("Loss",loss_value,step=cur_stp)
      tf.summary.scalar("rpn_cross_entropy loss",loss.loss_detail["rpn_cross_entropy"],step=cur_stp)
      tf.summary.scalar("rpn_loss_box loss",loss.loss_detail["rpn_loss_box"],step=cur_stp)
      tf.summary.scalar("cross_entropy loss",loss.loss_detail["cross_entropy"],step=cur_stp)
      tf.summary.scalar("loss_box loss",loss.loss_detail["loss_box"],step=cur_stp)

      cur_rpn_cross_entropy = loss.loss_detail["rpn_cross_entropy"]
      cur_rpn_loss_box = loss.loss_detail["rpn_loss_box"]
      cur_cross_entropy = loss.loss_detail["cross_entropy"]
      cur_loss_box = loss.loss_detail["loss_box"]
      cur_loss = loss_value

      grads = tape.gradient(loss_value, model.trainable_variables)
      g_rpn_cross_entropy = tape.gradient(loss.loss_detail["rpn_cross_entropy"], model.trainable_variables)
      g_rpn_loss_box = tape.gradient(loss.loss_detail["rpn_loss_box"], model.trainable_variables)
      g_cross_entropy = tape.gradient(loss.loss_detail["cross_entropy"], model.trainable_variables)
      g_loss_box = tape.gradient(loss.loss_detail["loss_box"], model.trainable_variables)

      opt.apply_gradients(zip(g_rpn_cross_entropy, model.trainable_variables))
      opt.apply_gradients(zip(g_rpn_loss_box, model.trainable_variables))

    tend = datetime.now() - tstart
    print("======================================")
    print("Batch {}, setp {} ==>> {}.".format(self.batch+1,self.current_step+1,cur_stp+1))
    print("Current Loss: {}.".format(cur_loss))
    print("Current cur_rpn_cross_entropy = {}.".format(cur_rpn_cross_entropy))
    print("Current cur_rpn_loss_box = {}.".format(cur_rpn_loss_box))
    print("Current cur_cross_entropy = {}.".format(cur_cross_entropy))
    print("Current cur_loss_box = {}.".format(cur_loss_box))
    print("Time usage: {} Day {} Second".format(tend.days,tend.seconds))
    self.current_step = cur_stp
    self.batch += 1