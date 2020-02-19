import os, sys
import tensorflow as tf
import numpy as np
import argparse
from datetime import datetime
from mydataset.ctw import CTW
from mydataset.svt import SVT
from lib.model.config import cfg
from lib.model.faster_rcnn import Faster_RCNN, RCNNLoss
from lib.model.label_rcnn import Label_RCNN, LRCNNLoss
from lib.frcnn_trainer import FRCNNTrainer
from lib.label_rcnn_trainer import LRCNNTrainer
from lib.tflib.evaluate_tools import draw_boxes
from lib.tflib.log_tools import save_image

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Choose settings.')
  parser.add_argument('--proposal', help='Choose proposal in nms and top_k.',default='top_k')
  parser.add_argument('--opt', help='Choose optimizer in sgd and adam.',default='adam')
  parser.add_argument('--debug', help='Set --debug if want to debug.', action="store_true")
  parser.add_argument('--net', help='Choose noework (frcnn/lrcnn).', default="lrcnn")
  parser.add_argument('--name', help='Name of task.')
  parser.add_argument('--dataset', help='Choose dataset.', default="svt")
  parser.add_argument('--datax', type=int, help='Dataset output width.',default=1280)
  parser.add_argument('--datay', type=int, help='Dataset output height.',default=720)
  parser.add_argument('--step', type=int, help='Step size.',default=10)
  parser.add_argument('--batch', type=int, help='Batch size.',default=20)
  parser.add_argument('--cross', help='Set --cross if want to cross box loss.', action="store_true")
  # parser.add_argument('--savestep', type=int, help='Batch size.',default=20)
  parser.add_argument('--learnrate', type=float, help='Learning rate.',default=0.001)
  args = parser.parse_args()
  time_start = datetime.now()
  print("Start when {}.\n".format(time_start.strftime("%Y%m%d-%H%M%S")))
  print("Running with: \n\t Use proposal: {},\n\t Is debug: {}.\n".format(args.proposal,args.debug))
  print("\t Step size: {},\n\t Batch size: {}.\n".format(args.step,args.batch))
  print("\t Data size: {} X {}.\n".format(args.datax,args.datay))
  print("\t Optimizer: {}.\n".format(args.opt))
  print("\t Taks name: {}.\n".format(args.name))
  print("\t Use cross: {}.\n".format(args.cross))

  isdebug = args.debug
  # isdebug = True
  
  if(args.opt.lower()=='sgd'):
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learnrate)
  else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learnrate)

  if(args.dataset=='svt'):
    mydatalog = SVT(out_size=[args.datax,args.datay])
    gtformat = 'xywh'
  else:
    mydatalog = CTW(out_size=[args.datax,args.datay])
    gtformat='yxyx'

  if(args.net=='frcnn'):
    # faster RCNN
    tkname = "{}_with_{}_{}_{}".format(args.name,args.proposal,args.dataset,args.opt) if(args.name!=None) else "FRCNN_with_{}_{}_{}".format(args.proposal,args.dataset,args.opt)
    trainer = FRCNNTrainer(isdebug=isdebug,task_name=tkname)
    model = Faster_RCNN(num_classes=2,bx_choose=args.proposal)
    loss = RCNNLoss(cfg=cfg,cfg_name="TRAIN",gtformat=gtformat)
  else:
    # label RCNN
    tkname = "{}_with_{}_{}".format(args.name,args.dataset,args.opt) if(args.name!=None) else "LRCNN_with_{}_{}".format(args.dataset,args.opt)
    trainer = LRCNNTrainer(isdebug=isdebug,task_name=tkname,gtformat=gtformat)
    model = Label_RCNN(num_classes=2)
    loss = LRCNNLoss(imge_size=[args.datay,args.datax],gtformat=gtformat)
  
  if(not(isdebug)):
    last_model = trainer.load(model)
  if(not(isdebug) and last_model!=None):
    model = last_model
    mydatalog.setconter(trainer.data_count)

  model.compile(
    optimizer=optimizer,
    loss=loss,
    )
  
  if(isdebug):
    for i in range(10):
      x_train, y_train = mydatalog.read_train_batch(3)
      sta = trainer.fit(x_train,y_train,model,loss,optimizer)
      if(sta==-1):
        break
  else:
    islog=False
    for i in range(args.batch):
      x_train, y_train = mydatalog.read_train_batch(args.step)
      # x_val, y_val = mydatalog.read_test_batch(2)
      x_val, y_val = x_train[0:3], y_train[0:3]
      # if(islog==False):
      #   imgs = draw_boxes(x_train,y_train)
      #   # imgs = tf.split(imgs,imgs.shape[0],axis=0)
      #   # for i in range(len(imgs)):
      #   #   save_image(imgs[i],'/home/yomcoding/TensorFlow/FasterRCNN/log/x_train_demo_{}.jpg'.format(i))
      #   trainer.log_image(imgs,10,name="{} training data examples.".format(imgs.shape[0]))
      #   islog=True
        
      trainer.fit(x_train,y_train,model,loss,optimizer)
      if(i<11):
        trainer.evaluate(x_val,y_val)
        # trainer.evaluate(x_train[0:2],y_val[0:2])
      if(i==int(args.batch/2)):
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.learnrate)
        trainer.set_trainer(opt=optimizer)
      # if(i%10==0):
      #   trainer.set_trainer(data_count=mydatalog._init_conter)
      #   trainer.save()
  
  time_usage = datetime.now()
  print("End at: {}.\n".format(time_usage.strftime("%Y%m%d-%H%M%S")))
  time_usage = time_usage - time_start
  print("Time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))

