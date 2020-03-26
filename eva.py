import os, sys
import tensorflow as tf
import numpy as np
import argparse
from lib.dataloader.ctw import CTW
from lib.dataloader.svt import SVT
from lib.dataloader.total import TTText
from lib.model.config import cfg
from lib.model.faster_rcnn import Faster_RCNN, RCNNLoss
from lib.model.label_rcnn import Label_RCNN, LRCNNLoss
from lib.frcnn_trainer import FRCNNTrainer
from lib.label_rcnn_trainer import LRCNNTrainer

__DEF_LOCAL_DIR = os.path.split(__file__)[0]
__DEF_MODEL_PATH = os.path.join(__DEF_LOCAL_DIR,"save_model")
__DEF_DATA_DIR = os.path.join(__DEF_LOCAL_DIR,'mydataset')
__DEF_CTW_DIR = os.path.join(__DEF_DATA_DIR,'ctw')
__DEF_SVT_DIR = os.path.join(__DEF_DATA_DIR,'svt')
__DEF_TTT_DIR = os.path.join(__DEF_DATA_DIR,'totaltext')


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Choose settings.')
  parser.add_argument('--net', help='Choose noework (frcnn/lrcnn).', default="lrcnn")
  parser.add_argument('--dataset', help='Choose dataset.', default="svt")
  parser.add_argument('--step', type=int, help='Step size.',default=5)
  parser.add_argument('--batch', type=int, help='Batch size.',default=20)
  parser.add_argument('--tname', help='Name of task, same with model dir.')

  args = parser.parse_args()
  if(not(os.path.exists(__DEF_MODEL_PATH))):sys.exit("Can't find save_model dir.")
  dir_list = os.listdir(__DEF_MODEL_PATH)
  if(len(dir_list)==0):sys.exit("Empty save_model dir.")
  tname = args.tname if(args.tname!=None and args.tname in dir_list)else dir_list[0]

  if(args.dataset.lower()=='svt'):
    gtformat = 'xywh'
  elif(args.dataset.lower()=='ttt'):
    gtformat = 'mask'
  else:
    gtformat='yxyx'

  if(args.net=='frcnn'):
    # faster RCNN
    trainer = FRCNNTrainer(isdebug=False,task_name=tname,logs_path = os.path.join(__DEF_LOCAL_DIR,'elog'))
    model = Faster_RCNN(num_classes=2,bx_choose=args.proposal)
  else:
    # label RCNN
    trainer = LRCNNTrainer(isdebug=False,task_name=tname,gtformat=gtformat,logs_path = os.path.join(__DEF_LOCAL_DIR,'elog'))
    model = Label_RCNN(num_classes=2)

  model = trainer.load(model,tname)
  if(model==None):sys.exit('Faild to load model in {}'.format(tname))
  
  print("Task {} start.\n".format(tname))
  print("\t Step size: {},\n\t Batch size: {}.\n".format(args.step,args.batch))
  print("\t Data size: {} X {}.\n".format(model.imgw,model.imgh))

  if(args.dataset.lower()=='svt'):
    mydatalog = SVT(__DEF_SVT_DIR,out_size=[model.imgh,model.imgw])
  elif(args.dataset.lower()=='ttt'):
    mydatalog = TTText(__DEF_TTT_DIR,out_size=[model.imgh,model.imgw])
  else:
    mydatalog = CTW(out_size=[model.imgh,model.imgw],
      ctw_dir=__DEF_CTW_DIR,
      train_img_dir=os.path.join(__DEF_CTW_DIR, 'train'),
      test_img_dir=os.path.join(__DEF_CTW_DIR, 'test'),
      ann_filename='ctw-annotations',
      log_dir=os.path.join(__DEF_CTW_DIR, 'log.txt'),
      )

  if(model!=None):
    x_test, y_test = mydatalog.read_test_batch(10)
    trainer.evaluate(x_test, y_test)
  

  
