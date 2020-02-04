import os, sys
import tensorflow as tf
import numpy as np
from model.config import cfg
from model.faster_rcnn import Faster_RCNN, RCNNLoss
from tflib.log_tools import _str2time, _str2num, _auto_scalar, _auto_image
from tflib.evaluate_tools import draw_boxes
from datetime import datetime

PROJ_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
LOGS_PATH = os.path.join(PROJ_PATH,"log")
MODEL_PATH = os.path.join(PROJ_PATH,"save_model")

def _chech_nan(tar):
  if(type(tar)==list):
    inc_list = []
    for i in range(len(tar)):
      if(tar[i]==None):
        continue
      inc=tf.where(tf.math.is_nan(tar[i]))
      if(inc.shape[0]!=0):
        inc_list.append([i,inc.shape[0]])
    if(len(inc_list)!=0):
      return inc_list
  else:
    inc=tf.where(tf.math.is_nan(tar))
    if(inc.shape[0]!=0):
      return inc.shape[0]
  return 0

class Trainer():
  def __init__(self,  
    logs_path = LOGS_PATH,
    model_path = MODEL_PATH,
    task_name = None,
    isdebug = False,
  ):
    """
    Args:
      logs_path: Tensor board dir.
      task_name: Floder name under logs_path.

    """
    super(Trainer,self).__init__()
    self.logs_path = logs_path if task_name == None else os.path.join(logs_path,task_name)
    self.model_path = model_path if task_name == None else os.path.join(model_path,task_name)
    self.logs_path = os.path.join(self.logs_path,datetime.now().strftime("%Y%m%d-%H%M%S"))
    if(not(os.path.exists(self.logs_path))):
      os.makedirs(self.logs_path)
    self.isdebug = isdebug if isdebug else False
    if(self.isdebug):
      self.file_writer = None
    else:
      self.file_writer = tf.summary.create_file_writer(self.logs_path)
      self.file_writer.set_as_default()
    self.grad_dict = {}
    self.loss_dict = {}
    self.current_step = 0
    self.batch = 0
    self.data_count = 0
    self.model = None
    self.loss = None
    self.opt = None
    
  def log_image(self, tfimg, log_num=10, img_size=None):
    if(self.isdebug and self.file_writer==None):
      self.file_writer = tf.summary.create_file_writer(self.logs_path)
      self.file_writer.set_as_default()

    with self.file_writer.as_default():
    # Don't forget to reshape.
      if(img_size!=None):
        tfimg = tf.reshape(tfimg, (-1, img_size[0], img_size[1], tfimg.shape[-1]))
      tf.summary.image("25 training data examples", tfimg, max_outputs=log_num, step=0)
  
  def set_trainer(self,model=None,loss=None,opt=None,data_count=None):
    if(model!=None):
      self.model = model
    if(loss!=None):
      self.loss = loss
    if(opt!=None):
      self.opt = opt
    if(data_count!=None):
      self.data_count = data_count

  def train_action(self,x_single,y_single):
    raise NotImplementedError
  
  def log_selfs(self):
    raise NotImplementedError

  def fit(self,
    x_train,y_train,
    model=None,loss=None,opt=None,
    x_val=None,y_val=None,
  ):
    tstart = datetime.now()
    total_data = x_train.shape[0]
    cur_stp = self.current_step

    logger = open(os.path.join(self.logs_path,'result.txt'),'a+',encoding='utf8')
    if(type(y_train)==list):
      assert(total_data==len(y_train))

    if(model!=None):
      self.model = model
    else:
      if(self.model!=None):
        model = self.model
      else:
        return

    if(loss!=None):
      self.loss = loss
    else:
      if(self.loss!=None):
        loss = self.loss
      else:
        return

    if(opt!=None):
      self.opt = opt
    else:
      if(self.opt!=None):
        opt = self.opt
      else:
        return
    try:
      type(model.trainable_variables)
    except:
      model(tf.zeros([1,]+x_train.shape[1:],dtype=tf.float32))

    if(total_data>1):
      x_train = tf.split(x_train,total_data,axis=0)
      cur_rpn_cross_entropy = 0.0
      cur_rpn_loss_box = 0.0
      cur_cross_entropy = 0.0
      cur_loss_box = 0.0
      cur_loss = 0.0
      cur_gtbox_num = 0.0

      for step in range(total_data):
        if(x_train[step].dtype!=tf.float32 or x_train[step].dtype!=tf.float64):
          x_train[step] = tf.cast(x_train[step],tf.float32)
        with tf.GradientTape(persistent=True) as tape:
          tape.watch(model.trainable_variables)
          y_pred = model(x_train[step])
          loss_value = loss(y_train[step], y_pred)

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
          _auto_scalar(loss_value,cur_stp+step,"Loss")
          _auto_scalar(loss.loss_detail,cur_stp+step)
          _auto_scalar(gtbox_num, cur_stp+step, "GT_box_num_pred")
          _auto_scalar(y_train[step].shape[0], cur_stp+step, "GT_box_num_true")
          _auto_scalar(gtbox_num / int(y_train[step].shape[0]), cur_stp+step, "GT_box_num_pred_div_true")
          if(gtbox_num>0 and int(y_train[step].shape[0]*2)>gtbox_num):
            bximg = draw_boxes(x_train[step],bbx)
            _auto_image(bximg,
              name="boxed_images in step {}".format(cur_stp+step),
              description="img in step {}".format(cur_stp+step))

        cur_rpn_cross_entropy += loss.loss_detail["rpn_cross_entropy"]
        cur_rpn_loss_box += loss.loss_detail["rpn_loss_box"]
        cur_cross_entropy += loss.loss_detail["cross_entropy"]
        cur_loss_box += loss.loss_detail["loss_box"]
        cur_loss += loss_value
        cur_gtbox_num += gtbox_num

        grads = tape.gradient(loss_value, model.trainable_variables)
        if(True):
          had_nan = False
          for iname in loss.loss_detail:
            self.grad_dict[iname]=tape.gradient(loss.loss_detail[iname], model.trainable_variables)
            nan_ind = _chech_nan(self.grad_dict[iname])
            if(type(nan_ind)==list or nan_ind!=0):
              logger.write("======================================\n")
              logger.write("Get NAN at batch {} setp {}.\n".format(self.batch+1,cur_stp+step))
              logger.write("From loss: {}, loss value: {}.\n".format(iname,loss.loss_detail[iname]))
              for iid in nan_ind:
                logger.write("\tGradient by {} has {} Nan.\n".format(model.trainable_variables[iid[0]].name,iid[1]))
              if(iname!="loss_box" and iname!="cross_entropy"):
                had_nan = True
          if(had_nan):
            logger.close()
            return -1
        
        if(self.isdebug):
          
          opt.apply_gradients(zip(self.grad_dict["rpn_cross_entropy"], model.trainable_variables))
          opt.apply_gradients(zip(self.grad_dict["rpn_loss_box"], model.trainable_variables))
          opt.apply_gradients(zip(self.grad_dict["rpn_cross_entropy1"], model.trainable_variables))
          opt.apply_gradients(zip(self.grad_dict["rpn_loss_box1"], model.trainable_variables))
          # opt.apply_gradients(zip(grads, model.trainable_variables))
        else:
          opt.apply_gradients(zip(self.grad_dict["rpn_cross_entropy"], model.trainable_variables))
          opt.apply_gradients(zip(self.grad_dict["rpn_loss_box"], model.trainable_variables))
          opt.apply_gradients(zip(self.grad_dict["rpn_cross_entropy1"], model.trainable_variables))
          opt.apply_gradients(zip(self.grad_dict["rpn_loss_box1"], model.trainable_variables))
          # opt.apply_gradients(zip(grads, model.trainable_variables))

      cur_stp += step
      cur_rpn_cross_entropy /= total_data
      cur_rpn_loss_box /= total_data
      cur_cross_entropy /= total_data
      cur_loss_box /= total_data
      cur_loss /= total_data
      cur_gtbox_num /= total_data

    else:
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        y_pred = model(x_train)
        loss_value = loss(y_train, y_pred)

      if(not(self.isdebug)):
        _auto_scalar(loss_value,cur_stp,"Loss")
        _auto_scalar(loss.loss_detail,cur_stp)

      grads = tape.gradient(loss_value, model.trainable_variables)
      if(True):
        g_rpn_cross_entropy = tape.gradient(loss.loss_detail["rpn_cross_entropy"], model.trainable_variables)
        g_rpn_loss_box = tape.gradient(loss.loss_detail["rpn_loss_box"], model.trainable_variables)
        g_rpn_cross_entropy1 = tape.gradient(loss.loss_detail["rpn_cross_entropy1"], model.trainable_variables)
        g_rpn_loss_box1 = tape.gradient(loss.loss_detail["rpn_loss_box1"], model.trainable_variables)
        g_cross_entropy = tape.gradient(loss.loss_detail["cross_entropy"], model.trainable_variables)
        g_loss_box = tape.gradient(loss.loss_detail["loss_box"], model.trainable_variables)
      if(self.isdebug):
        opt.apply_gradients(zip(g_rpn_cross_entropy, model.trainable_variables))
        opt.apply_gradients(zip(g_rpn_loss_box, model.trainable_variables))
      else:
        # opt.apply_gradients(zip(grads, model.trainable_variables))
        opt.apply_gradients(zip(g_rpn_cross_entropy, model.trainable_variables))
        opt.apply_gradients(zip(g_rpn_loss_box, model.trainable_variables))
        opt.apply_gradients(zip(g_rpn_cross_entropy1, model.trainable_variables))
        opt.apply_gradients(zip(g_rpn_loss_box1, model.trainable_variables))

      opt.apply_gradients(zip(grads, model.trainable_variables))
      cur_stp += 1

    tend = datetime.now() - tstart
    logger.write("======================================\n")
    logger.write("Batch {}, setp {} ==>> {}.\n".format(self.batch+1,self.current_step+1,cur_stp+1))
    logger.write("Avg Loss: {}.\n".format(cur_loss))
    logger.write("Avg cur_rpn_cross_entropy = {}.\n".format(cur_rpn_cross_entropy))
    logger.write("Avg cur_rpn_loss_box = {}.\n".format(cur_rpn_loss_box))
    logger.write("Avg cur_cross_entropy = {}.\n".format(cur_cross_entropy))
    logger.write("Avg cur_loss_box = {}.\n".format(cur_loss_box))
    logger.write("Avg cur_gtbox_num = {}.\n".format(cur_gtbox_num))
    logger.write("Time usage: {} Day {} Second.\n".format(tend.days,tend.seconds))
    logger.write("======================================\n\n")
    self.current_step = cur_stp
    self.batch += 1
    logger.close()
    return 0

  def save(self,model=None):
    self.set_trainer(model)
    if(self.model==None or self.loss==None or self.opt==None):
      return
    now_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(self.model_path,now_time)
    self.model.save_weights(os.path.join(save_path,'model'))
    # tf.saved_model.save(self.model,save_path)
    txtlog = open(os.path.join(save_path,'log.txt'),'a+',encoding='utf8')
    txtlog.write("Step = {} , Batch = {} , Data count = {} .".format(self.current_step,self.batch,self.data_count))

  def load(self,model,loddir=None):
    if(loddir==None):
      last_time = None
      if(not(os.path.exists(self.model_path))):
        return None
      for tardir in os.listdir(self.model_path):
        cur_time = _str2time(tardir)
        if(last_time==None or cur_time>last_time):
          last_time = cur_time
      if(last_time==None):
        return None
      loddir = os.path.join(self.model_path,last_time.strftime("%Y%m%d-%H%M%S"))
    try:
      model.load_weights(os.path.join(loddir,'model'))
    except Exception as e:
      print(str(e))
      return None
    self.set_trainer(model)

    try:
      txtlog = open(os.path.join(loddir,'log.txt'),'r',encoding='utf8')
      self.current_step,self.batch,self.data_count = _str2num(txtlog.readline())
      txtlog.close()
    except Exception as e:
      print(str(e))

    return model
    