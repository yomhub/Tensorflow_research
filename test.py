import os, sys
import tensorflow as tf
import numpy as np
import argparse
from datetime import datetime
from lib.dataloader.ctw import CTW
from lib.dataloader.svt import SVT
from lib.dataloader.total import TTText
from lib.model.config import cfg
from lib.model.faster_rcnn import Faster_RCNN, RCNNLoss
from lib.model.label_rcnn import Label_RCNN, LRCNNLoss
from lib.frcnn_trainer import FRCNNTrainer
from lib.label_rcnn_trainer import LRCNNTrainer
from lib.tflib.evaluate_tools import draw_boxes
from lib.tflib.log_tools import save_image

__DEF_INDEX = 2
__DEF_IMG_SIZE = [[1280,720],[int(1280/2),int(720/2)],[640,640]]

__DEF_LOCAL_DIR = os.path.split(__file__)[0]
__DEF_DATA_DIR = os.path.join(__DEF_LOCAL_DIR,'mydataset')
__DEF_CTW_DIR = os.path.join(__DEF_DATA_DIR,'ctw')
__DEF_SVT_DIR = os.path.join(__DEF_DATA_DIR,'svt')
__DEF_TTT_DIR = os.path.join(__DEF_DATA_DIR,'totaltext')


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Choose settings.')
  parser.add_argument('--proposal', help='Choose proposal in nms and top_k.',default='top_k')
  parser.add_argument('--opt', help='Choose optimizer in sgd and adam.',default='adam')
  parser.add_argument('--debug', help='Set --debug if want to debug.', action="store_true")
  parser.add_argument('--save', help='Set --save if want to save network.', action="store_true")
  parser.add_argument('--load', help='Set --load if want to load network.', action="store_true")
  parser.add_argument('--net', help='Choose noework (frcnn/lrcnn).', default="lrcnn")
  parser.add_argument('--name', help='Name of task.')
  parser.add_argument('--dataset', help='Choose dataset: ctw/svt/ttt.', default="ttt")
  parser.add_argument('--datax', type=int, help='Dataset output width.',default=__DEF_IMG_SIZE[__DEF_INDEX][0])
  parser.add_argument('--datay', type=int, help='Dataset output height.',default=__DEF_IMG_SIZE[__DEF_INDEX][1])
  parser.add_argument('--step', type=int, help='Step size.',default=10)
  parser.add_argument('--batch', type=int, help='Batch size.',default=20)
  parser.add_argument('--cross', help='Set --cross if want to cross box loss.', action="store_true")
  # parser.add_argument('--savestep', type=int, help='Batch size.',default=20)
  parser.add_argument('--learnrate', type=float, help='Learning rate.',default=0.007)
  args = parser.parse_args()
  time_start = datetime.now()
  print("Start when {}.\n".format(time_start.strftime("%Y%m%d-%H%M%S")))
  print("Running with: \n\t Use proposal: {},\n\t Is debug: {}.".format(args.proposal,args.debug))
  print("\t Step size: {},\n\t Batch size: {}.".format(args.step,args.batch))
  print("\t Data size: {} X {}.".format(args.datax,args.datay))
  print("\t Optimizer: {}.".format(args.opt))
  print("\t Taks name: {}.".format(args.name))
  print("\t Use cross: {}.".format(args.cross))
  print("\t Save network: {}.".format('Yes' if(args.save)else 'No'))
  print("\t Load network: {}.".format('Yes' if(args.load)else 'No'))

  isdebug = args.debug
  # isdebug = True
  opt_schedule = np.array([0.6,0.8])*args.batch
  opt_schedule = opt_schedule.astype(np.int)
  opt_names = ['sgd']
  if(args.opt.lower()=='sgd'):
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learnrate)
  else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learnrate)

  if(args.dataset.lower()=='svt'):
    mydatalog = SVT(__DEF_SVT_DIR,out_size=[args.datay,args.datax])
    gtformat = 'xywh'
  elif(args.dataset.lower()=='ttt'):
    mydatalog = TTText(__DEF_TTT_DIR,out_size=[args.datay,args.datax])
    gtformat = 'mask'
  else:
    mydatalog = CTW(out_size=[args.datay,args.datax],
      ctw_dir=__DEF_CTW_DIR,
      train_img_dir=os.path.join(__DEF_CTW_DIR, 'train'),
      test_img_dir=os.path.join(__DEF_CTW_DIR, 'test'),
      ann_filename='ctw-annotations',
      log_dir=os.path.join(__DEF_CTW_DIR, 'log.txt'),
      )
    gtformat='yxyx'
    

  if(args.net=='frcnn'):
    # faster RCNN
    tkname = "{}_with_{}_{}_{}".format(args.name,args.proposal,args.dataset,args.opt) if(args.name!=None) else "FRCNN_with_{}_{}_{}".format(args.proposal,args.dataset,args.opt)
    trainer = FRCNNTrainer(isdebug=isdebug,task_name=tkname)
    model = Faster_RCNN(num_classes=2,bx_choose=args.proposal)
    loss = RCNNLoss(cfg=cfg,cfg_name="TRAIN",gtformat=gtformat)
  else:
    # label RCNN
    dif_nor = False
    tkname = "{}_with_{}_{}".format(args.name,args.dataset,args.opt) if(args.name!=None) else "LRCNN_with_{}_{}".format(args.dataset,args.opt)
    trainer = LRCNNTrainer(isdebug=isdebug,task_name=tkname,gtformat=gtformat)
    model = Label_RCNN(num_classes=2,dif_nor=dif_nor)
    loss = LRCNNLoss(imge_size=[args.datay,args.datax],gtformat=gtformat,dif_nor=dif_nor)
  
  if(not(isdebug) and args.load):
    last_model = trainer.load(model)
    if(last_model!=None):
      model = last_model
      mydatalog.setconter(trainer.data_count)

  model.compile(
    optimizer=optimizer,
    loss=loss,
    )

  if(isdebug):
    for i in range(3):
      x_train, y_train = mydatalog.read_train_batch(3)
      ret = trainer.fit(x_train,y_train,model,loss,optimizer)
      if(ret==-1):
        break
  else:
    islog=False
    # loss.gtformat='xywh'
    # init_data = SVT(__DEF_SVT_DIR,out_size=[args.datay,args.datax])
    trainer.set_trainer(model=model,loss=loss,opt=optimizer)

    # for i in range(3):
    #   x_train, y_train = init_data.read_train_batch(3)
    #   trainer.fit(x_train,y_train)
    # loss.gtformat='mask'
    # optimizer = tf.keras.optimizers.Adam(learning_rate=args.learnrate)
    # trainer.set_trainer(loss=loss,opt=optimizer)
    for i in range(args.batch):
      x_train, y_train = mydatalog.read_train_batch(args.step)
      x_val, y_val = mydatalog.read_test_batch(2)
      # x_val, y_val = x_train[0:2], y_train[0:2]
      # if(islog==False):
      #   imgs = draw_boxes(x_train,y_train)
      #   # imgs = tf.split(imgs,imgs.shape[0],axis=0)
      #   # for i in range(len(imgs)):
      #   #   save_image(imgs[i],'/home/yomcoding/TensorFlow/FasterRCNN/log/x_train_demo_{}.jpg'.format(i))
      #   trainer.log_image(imgs,10,name="{} training data examples.".format(imgs.shape[0]))
      #   islog=True
        
      ret = trainer.fit(x_train,y_train)
      if(ret==-1):break
      trainer.evaluate(x_val,y_val)
      trainer.evaluate(x_train[0:2],y_train[0:2])
      # inc = np.where(opt_schedule==i)
      if(i==5):
        trainer.set_trainer(opt=tf.keras.optimizers.Adam(learning_rate=0.005))
      if(i==10):
        trainer.set_trainer(opt=tf.keras.optimizers.Adam(learning_rate=0.001))
      # if(i%10==0):
      #   trainer.set_trainer(data_count=mydatalog._init_conter)
      #   trainer.save()
  if(ret==0 and args.save):
    trainer.set_trainer(data_count=mydatalog.train_conter)
    trainer.save()
  time_usage = datetime.now()
  print("End at: {}.\n".format(time_usage.strftime("%Y%m%d-%H%M%S")))
  time_usage = time_usage - time_start
  print("Time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))

