import os, sys
import tensorflow as tf
import numpy as np
import argparse
from mydataset.ctw import CTW
from mydataset.svt import SVT
from lib.model.config import cfg
from lib.model.faster_rcnn import Faster_RCNN, RCNNLoss
from lib.trainer import Trainer

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Choose settings.')
  parser.add_argument('--proposal', help='Choose proposal in nms and top_k.',default='top_k')
  parser.add_argument('--debug', help='Set --debug if want to debug.', action="store_true")
  parser.add_argument('--dataset', help='Choose dataset.', default="svt")
  parser.add_argument('--datax', type=int, help='Dataset output width.',default=1024)
  parser.add_argument('--datay', type=int, help='Dataset output height.',default=1024)
  parser.add_argument('--step', type=int, help='Step size.',default=50)
  parser.add_argument('--batch', type=int, help='Batch size.',default=20)
  # parser.add_argument('--savestep', type=int, help='Batch size.',default=20)
  parser.add_argument('--learnrate', type=float, help='Learning rate.',default=0.001)
  args = parser.parse_args()

  print("Running with: \n\t Use proposal: {},\n\t Is debug: {}.\n".format(args.proposal,args.debug))
  print("\t Step size: {},\n\t Batch size: {}.\n".format(args.step,args.batch))
  
  isdebug = args.debug
  isdebug = True
  learning_rate = args.learnrate

  if(args.dataset=='svt'):
    mydatalog = SVT(out_size=[args.datax,args.datay])
    loss = RCNNLoss(cfg=cfg,cfg_name="TRAIN",gtformat='xywh')
  else:
    loss = RCNNLoss(cfg=cfg,cfg_name="TRAIN",gtformat='yxyx')
    mydatalog = CTW(out_size=[args.datax,args.datay])

  model = Faster_RCNN(num_classes=2,bx_choose=args.proposal)
  if(not(isdebug)):
    last_model = trainer.load(model)
  if(not(isdebug) and last_model!=None):
    model = last_model
    mydatalog.setconter(trainer.data_count)

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  trainer = Trainer(isdebug=isdebug,task_name="{}_with_{}".format(args.proposal,args.dataset))

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
    print(type(args.batch))
    for i in range(args.batch):
      x_train, y_train = mydatalog.read_train_batch(args.step)
      trainer.fit(x_train,y_train,model,loss,optimizer)
      
      if(i%4==0):
        trainer.set_trainer(data_count=mydatalog._init_conter)
        trainer.save()

  print()

