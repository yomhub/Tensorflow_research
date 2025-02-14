import os, sys
import tensorflow as tf
import numpy as np
from tflib.log_tools import str2time, str2num, auto_scalar, auto_image
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
    self.isdebug = isdebug if isdebug else False
    self.logs_path = logs_path if task_name == None else os.path.join(logs_path,task_name)
    self.model_path = model_path
    if(not(self.isdebug)):
      self.logs_path = os.path.join(self.logs_path,datetime.now().strftime("%Y%m%d-%H%M%S"))
    if(not(os.path.exists(self.logs_path))):
      os.makedirs(self.logs_path)
    if(self.isdebug):
      self.file_writer = None
      os.makedirs(os.path.join(self.logs_path,'debug'),exist_ok=True)
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
    self.eva_step = 0
    self.task_name = task_name
    
  def log_txt(self,logstr):
    if(logstr and type(logstr)==str):
      if(self.isdebug):logger = open(os.path.join(self.logs_path,'debug','result.txt'),'a+',encoding='utf8')
      else:logger = open(os.path.join(self.logs_path,'result.txt'),'a+',encoding='utf8')
      logger.write(logstr)

  def log_image(self, tfimg, log_num=10, img_size=None, name=None):
    if(self.isdebug and self.file_writer==None):
      self.file_writer = tf.summary.create_file_writer(self.logs_path)
      self.file_writer.set_as_default()
    log_num = log_num if log_num<tfimg.shape[0] else int(tfimg.shape[0])
    if(name == None):
      name = "{} images.".format(log_num)
    if(tf.reduce_max(tfimg)>1.0):
      tfimg = tfimg/256.0
    with self.file_writer.as_default():
      # # Don't forget to resize.
      # if(img_size!=None):
      #   tfimg = tf.image.resize(tfimg, img_size)
      tf.summary.image(name, tfimg, step=0, max_outputs=log_num)
  
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

  def eval_action(self,x_single,y_single,step,logger):
    raise NotImplementedError

  def eval_callback(self,total_size,logger,time_usage):
    raise NotImplementedError

  def evaluate(self,x_val,y_val,model=None):
    tstart = datetime.now()
    if(type(x_val)!=list):
      x_val = tf.split(x_val,x_val.shape[0],axis=0)
    total_data = len(x_val)
    
    if(type(y_val)!=list):
      y_val=[y_val]
    assert(total_data==len(y_val))

    logger = open(os.path.join(self.logs_path,'evaluate.txt'),'a+',encoding='utf8')
    if(not(self.isdebug)):
      logger.write(datetime.now().strftime("%Y%m%d-%H%M%S")+'\n')
    if(model!=None):
      self.model = model
    else:
      if(self.model!=None):
        model = self.model
      else:
        return
  
    cur_stp = self.eva_step
    for step in range(total_data):
      if(x_val[step].dtype!=tf.float32 or x_val[step].dtype!=tf.float64):
        x_val[step] = tf.cast(x_val[step],tf.float32)
      ret = self.eval_action(x_val[step],y_val[step],cur_stp,logger)
      cur_stp += 1
      if(ret==-1):
        logger.close()
        return ret
    self.eval_callback(total_data,logger,datetime.now() - tstart)
    self.eva_step = cur_stp
    logger.close()
    return 0

  def fit(self,
    x_train,y_train,
    model=None,loss=None,opt=None,
    x_val=None,y_val=None,
  ):
    tstart = datetime.now()
    if(type(x_train)!=list):
      x_train = tf.split(x_train,x_train.shape[0],axis=0)
    total_data = len(x_train)
    cur_stp = self.current_step
    logger = open(os.path.join(self.logs_path,'result.txt'),'a+',encoding='utf8')
    if(not(self.isdebug)):
      logger.write(datetime.now().strftime("%Y%m%d-%H%M%S")+'\n')
    if(type(y_train)!=list):
      y_train = [y_train]
    assert(total_data==len(y_train))

    if(self.model==None):
      if(model!=None):
        self.model = model
      else:
        return -1

    if(self.loss==None):
      if(loss!=None):
        self.loss = loss
      else:
        return -1

    if(self.opt==None):
      if(opt!=None):
        self.opt = opt
      else:
        return -1

    try:
      type(self.model.trainable_variables)
    except:
      self.model(tf.zeros([1,]+x_train[0].shape[1:],dtype=tf.float32))

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
    if(x_val!=None and y_val!=None):
      self.evaluate(x_val,y_val)
    return 0

  def save(self,model=None):
    self.set_trainer(model)
    if(self.model==None or self.loss==None or self.opt==None):
      return
    now_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(self.model_path,self.task_name,now_time)
    self.model.save_weights(os.path.join(save_path,'model'))
    # tf.saved_model.save(self.model,save_path)
    txtlog = open(os.path.join(save_path,'log.txt'),'a+',encoding='utf8')
    txtlog.write("Step = {} , Batch = {} , Data count = {} .".format(self.current_step,self.batch,self.data_count))

  def load(self,model,tsk_name=None):
    # get name
    if(not(os.path.exists(self.model_path))):return None
    tsk_list = os.listdir(self.model_path)
    if(len(tsk_list)==0):return None
    tsk_name = tsk_name if(tsk_name!=None and tsk_name in tsk_list)else tsk_list[0]
    # get time
    last_time = None
    for tardir in os.listdir(os.path.join(self.model_path,tsk_name)):
      cur_time = str2time(tardir)
      if(last_time==None or cur_time>last_time):
        last_time = cur_time
    if(last_time==None):
      return None
    # final dir
    loddir = os.path.join(self.model_path,tsk_name,last_time.strftime("%Y%m%d-%H%M%S"))

    # load config
    try:
      txtlog = open(os.path.join(loddir,'log.txt'),'r',encoding='utf8')
      self.current_step,self.batch,self.data_count = str2num(txtlog.readline())
      txtlog.close()
    except Exception as e:
      print(str(e))
      return None

    # model(tf.zeros([1,imgh,imgw,3],dtype=tf.float32))
    try:
      model.load_weights(os.path.join(loddir,'model'))
    except Exception as e:
      print(str(e))
      return None
    self.model = model

    return model
    