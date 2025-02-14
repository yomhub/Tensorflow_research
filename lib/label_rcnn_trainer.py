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

class LRCNNTrainer(Trainer):
  def __init__(self,task_name,isdebug,
    threshold=0.7,gtformat='yxyx',gen_box_by_gt=False,ol_score=False,draw_gt=False,
    logs_path = LOGS_PATH,
    model_path = MODEL_PATH,):
    Trainer.__init__(self,task_name=task_name,isdebug=isdebug,logs_path = logs_path,model_path = model_path,)
    self.cur_loss = 0.0
    self.threshold = threshold
    gtformat = gtformat.lower()
    self.gtformat = gtformat
    if(not(gtformat in ['yxyx','xywh','mask'])):self.gtformat='yxyx'
    self.gen_box_by_gt = gen_box_by_gt
    self.ol_score = ol_score
    self.draw_gt = bool(draw_gt)
    # super(FRCNNTrainer,self).__init__(kwargs)

  def train_action(self,x_single,y_single,step,logger):
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(self.model.trainable_variables)
      y_pred = self.model(x_single)
      loss_value = self.loss(y_single, y_pred)

    self.cur_loss += loss_value
    grads = tape.gradient(loss_value, self.model.trainable_variables)
    self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    if(not(self.isdebug)):
      hit = False
      auto_scalar(loss_value,step,"Loss")
      for itm in self.model.trainable_variables:
        auto_scalar(tf.reduce_mean(itm),step,itm.name)
      for itn in y_pred:
        auto_scalar(tf.reduce_mean(y_pred[itn]),step,itn)
      for iname in self.loss.loss_detail:
        auto_scalar(self.loss.loss_detail[iname],step,"Loss detail: {}".format(iname))
      for i in range(len(grads)):
        if(tf.math.is_nan(tf.reduce_mean(grads[i]))):
          logger.write("Nan in {} in step {}".format(self.model.trainable_variables[i].name,step))
          hit = True
          for iname in self.loss.loss_detail:
            logger.write("Gradient in loss {} is:".format(iname))
            logger.write(tf.reduce_mean(tape.gradient(self.loss.loss_detail[iname], self.model.trainable_variables))+'\n')
      if(hit):return -1
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

  def draw_gt_pred_box(self,score,bbox,recf_size,gt_box,image):
    imgh = float(image.shape[-3])
    imgw = float(image.shape[-2])
    feat_size = score.shape[-3:-1]
    bbox = tf.reshape(bbox,[-1,4])
    col = tf.random.uniform(
      (1,3),
      minval=128,
      maxval=256,
    )
    if(self.draw_gt):
      if(self.gtformat in ['xywh','yxyx']):
        # draw gt box frist
        image = draw_grid_in_gt(recf_size,gt_box,image)
      else:
        image = draw_msk_in_gt(gt_box,image)
      
    if(len(score.shape)==3):score = score[:,:,:(int(bbox.shape[-1]/4)+1)]
    elif(len(score.shape)==4):score = score[:,:,:,:(int(bbox.shape[-1]/4)+1)]
    score = tf.reshape(score,[-1,score.shape[-1]])
    score = tf.math.argmax(score,axis=-1)
    l1box = bbox[score>0]

    if(l1box.shape[0]!=None and l1box.shape[0]>0):
      # if we have box, draw it
      l1box = tf.stack([l1box[:,0]/imgh,l1box[:,1]/imgw,l1box[:,2]/imgh,l1box[:,3]/imgw],axis=1)
      image = tf.image.draw_bounding_boxes(image,tf.reshape(l1box,[1,-1,4]),col)
    if(self.gen_box_by_gt):
      if(self.gtformat in ['xywh','yxyx']):
        # or forcibly draw box in red color
        score = gen_label_from_gt(feat_size,recf_size,gt_box,[imgh,imgw])
        score = tf.reshape(score,[-1])
        l1box = tf.reshape(tf.gather(bbox,tf.where(score>0)),[-1,4])
        l1box = tf.stack([l1box[:,0]/imgh,l1box[:,1]/imgw,l1box[:,2]/imgh,l1box[:,3]/imgw],axis=1)
        image = tf.image.draw_bounding_boxes(image,tf.reshape(l1box,[1,-1,4]),tf.convert_to_tensor([[255.0,0.0,0.0]]))
      else:
        msk = gen_gt_from_msk(gt_box,feat_size,recf_size,score.shape[-1])
        msk = tf.reshape(msk,[-1])
        l1box = bbox[msk>0]
        l1box = tf.stack([l1box[:,0]/imgh,l1box[:,1]/imgw,l1box[:,2]/imgh,l1box[:,3]/imgw],axis=1)
        image = tf.image.draw_bounding_boxes(image,tf.reshape(l1box,[1,-1,4]),tf.convert_to_tensor([[255.0,0.0,0.0]]))
    return image

  def eval_action(self,x_single,y_single,step,logger):
    y_pred = self.model(x_single)
    # imgh = float(x_single.shape[-3])
    # imgw = float(x_single.shape[-2])
    if(self.gtformat=='xywh'):
      gt_box = xywh2yxyx(y_single[:,1:])
      label_gt = tf.stack([y_single[:,0],gt_box[:,0],gt_box[:,1],gt_box[:,2],gt_box[:,3]],axis=1)
    elif(self.gtformat=='yxyx'):
      gt_box = y_single[:,1:]
      label_gt = tf.stack([y_single[:,0],gt_box[:,0],gt_box[:,1],gt_box[:,2],gt_box[:,3]],axis=1)
    else:label_gt = y_single
    name_list = [o for o in y_pred if('bbox' in o)]
    # if(self.ol_score and self.gtformat in ['xywh','yxyx']):
    #   l1op = tf.reduce_sum(label_overlap_tf(gt_box,x_single.shape[1:3],gen_label_from_prob(y_pred["l1_score"])))
    #   l2op = tf.reduce_sum(label_overlap_tf(gt_box,x_single.shape[1:3],gen_label_from_prob(y_pred["l2_score"])))
    #   l3op = tf.reduce_sum(label_overlap_tf(gt_box,x_single.shape[1:3],gen_label_from_prob(y_pred["l3_score"])))
    #   auto_scalar(l1op,step,"L1 overlap")
    #   auto_scalar(l2op,step,"L2 overlap")
    #   auto_scalar(l3op,step,"L3 overlap")
      
    for o in name_list:
      lx = o[0:2]
      tmp = self.draw_gt_pred_box(y_pred[lx+"_score"],y_pred[lx+"_bbox"],y_pred[lx+"_rf_s"], label_gt, x_single)
      if(tf.reduce_max(tmp)>1.0):
        tmp = tmp/256.0
      tf.summary.image(name="Boxed image in {} in step {}.".format(lx.upper(),self.current_step),data=tmp,step=0,max_outputs=tmp.shape[0])

    return 0

  def eval_callback(self,total_size,logger,time_usage):
    if(logger.writable()):
      logger.write("======================================\n")
      logger.write("Time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))
    return 0

    
