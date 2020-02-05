import os, sys
import tensorflow as tf
import numpy as np
from tflib.log_tools import _str2time, _str2num, _auto_scalar, _auto_image
from tflib.evaluate_tools import draw_boxes
from datetime import datetime

PROJ_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
LOGS_PATH = os.path.join(PROJ_PATH,"log")
MODEL_PATH = os.path.join(PROJ_PATH,"save_model")

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
  
  def train_action(self,x_single,y_single,step,logger):
    raise NotImplementedError
  
  def log_action(self,logger):
    raise NotImplementedError

  def batch_callback(self,batch_size,logger,time_usage):
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

    x_train = tf.split(x_train,total_data,axis=0)
    for step in range(total_data):
      if(x_train[step].dtype!=tf.float32 or x_train[step].dtype!=tf.float64):
        x_train[step] = tf.cast(x_train[step],tf.float32)
      ret = self.train_action(x_train[step],y_train[step],cur_stp,logger)
      cur_stp += 1
      if(ret==-1):
        logger.close()
        return ret
    self.batch_callback(total_data,logger,datetime.now() - tstart)
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
    