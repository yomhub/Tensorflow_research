import os, sys
import tensorflow as tf
import numpy as np
import argparse
from mydataset.ctw import CTW
from mydataset.svt import SVT
from lib.model.config import cfg
from lib.model.faster_rcnn import Faster_RCNN, RCNNLoss
from lib.model.label_rcnn import Label_RCNN, LRCNNLoss
from lib.frcnn_trainer import FRCNNTrainer
from lib.label_rcnn_trainer import LRCNNTrainer

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Choose settings.')
  parser.add_argument('--net', help='Choose noework (frcnn/lrcnn).', default="lrcnn")
  parser.add_argument('--dataset', help='Choose dataset.', default="svt")
  parser.add_argument('--datax', type=int, help='Dataset output width.',default=1280)
  parser.add_argument('--datay', type=int, help='Dataset output height.',default=720)
  parser.add_argument('--step', type=int, help='Step size.',default=5)
  parser.add_argument('--batch', type=int, help='Batch size.',default=20)
  parser.add_argument('--tname', help='Name of task, same with model dir.', default="LRCNN_with_svt_adam")
  args = parser.parse_args()

  print("Task {} start.\n".format(args.tname))
  print("\t Step size: {},\n\t Batch size: {}.\n".format(args.step,args.batch))
  print("\t Data size: {} X {}.\n".format(args.datax,args.datay))

  if(args.dataset=='svt'):
    mydatalog = SVT(out_size=[args.datax,args.datay])
    gtformat = 'xywh'
  else:
    mydatalog = CTW(out_size=[args.datax,args.datay])
    gtformat='yxyx'

  if(args.net=='frcnn'):
    # faster RCNN
    trainer = FRCNNTrainer(isdebug=False,task_name=args.tname)
    model = Faster_RCNN(num_classes=2,bx_choose=args.proposal)
    loss = RCNNLoss(cfg=cfg,cfg_name="TRAIN",gtformat=gtformat)
  else:
    # label RCNN
    trainer = LRCNNTrainer(isdebug=False,task_name=args.tname,gtformat=gtformat)
    model = Label_RCNN(num_classes=2)
    loss = LRCNNLoss(imge_size=[args.datax,args.datay],gtformat=gtformat)
    
  model(tf.zeros([1,args.datay,args.datax,3],dtype=tf.float32))
  model = trainer.load(model)
  if(model!=None):
    x_test, y_test = mydatalog.read_test_batch(10)
    trainer.evaluate(x_test, y_test)
  

  
