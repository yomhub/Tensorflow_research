import os, sys
import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm, trange
from datetime import datetime
from lib.dataloader.ctw import CTW
from lib.dataloader.svt import SVT
from lib.dataloader.total import TTText
from lib.model.config import cfg
from lib.model.faster_rcnn import Faster_RCNN, RCNNLoss
from lib.model.label_rcnn import Label_RCNN, LRCNNLoss,Label_RCNN_v2, LRCNNLoss_v2
from lib.model.unet import Unet, UnetLoss
from lib.frcnn_trainer import FRCNNTrainer
from lib.label_rcnn_trainer import LRCNNTrainer
from lib.unet_trainer import UnetTrainer
from lib.tflib.evaluate_tools import draw_boxes
from lib.tflib.log_tools import save_image
from lib.config.train_default import cfg as tcfg

__DEF_LOCAL_DIR = os.path.split(__file__)[0]
__DEF_DATA_DIR = os.path.join(__DEF_LOCAL_DIR,'mydataset')
__DEF_CTW_DIR = os.path.join(__DEF_DATA_DIR,'ctw')
__DEF_SVT_DIR = os.path.join(__DEF_DATA_DIR,'svt')
__DEF_TTT_DIR = os.path.join(__DEF_DATA_DIR,'totaltext')
# sys.stderr = open('err.log','w')

def get_opt(oname,lr):
  if(oname=='sgd'):
    return tf.keras.optimizers.SGD(learning_rate=lr,momentum=tcfg['MMT'])
  else:
    return tf.keras.optimizers.Adam(learning_rate=lr)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Choose settings.')
  parser.add_argument('--proposal', help='Choose proposal in nms and top_k.',default='top_k')
  parser.add_argument('--opt', help='Choose optimizer.',default=tcfg['OPT'])
  parser.add_argument('--debug', help='Set --debug if want to debug.', action="store_true")
  parser.add_argument('--save', help='Set --save if want to save network.', action="store_true")
  parser.add_argument('--load', help='Set --load if want to load network.', action="store_true")
  parser.add_argument('--net', help='Choose noework (frcnn/lrcnn/unet).', default="unet")
  parser.add_argument('--name', help='Name of task.')
  parser.add_argument('--dataset', help='Choose dataset: ctw/svt/ttt.', default=tcfg['DATASET'])
  parser.add_argument('--datax', type=int, help='Dataset output width.',default=tcfg['IMG_SIZE'][0])
  parser.add_argument('--datay', type=int, help='Dataset output height.',default=tcfg['IMG_SIZE'][1])
  parser.add_argument('--step', type=int, help='Step size.',default=tcfg['STEP'])
  parser.add_argument('--batch', type=int, help='Batch size.',default=tcfg['BATCH'])
  parser.add_argument('--logstp', type=int, help='Log step size.',default=tcfg['LOGSTP'])
  parser.add_argument('--cross', help='Set --cross if want to cross box loss.', action="store_true")
  # parser.add_argument('--savestep', type=int, help='Batch size.',default=20)
  parser.add_argument('--learnrate', type=float, help='Learning rate.',default=tcfg['LR'])
  args = parser.parse_args()
  time_start = datetime.now()
  isdebug = args.debug
  lr = args.learnrate
  # isdebug = True

  summarize = "Start when {}.\n".format(time_start.strftime("%Y%m%d-%H%M%S")) +\
    "Running with: \n\t Use proposal: {},\n\t Is debug: {}.\n".format(args.proposal,args.debug)+\
    "\t Step size: {},\n\t Batch size: {}.\n".format(args.step,args.batch)+\
    "\t Data size: {} X {}.\n".format(args.datax,args.datay)+\
    "\t Optimizer: {}.\n".format(args.opt)+\
    "\t Init learning rate: {}.\n".format(lr)+\
    "\t Taks name: {}.\n".format(args.name)+\
    "\t Use cross: {}.\n".format(args.cross)+\
    "\t Save network: {}.\n".format('Yes' if(args.save)else 'No')+\
    "\t Load network: {}.\n".format('Yes' if(args.load)else 'No')
  print(summarize)
  
  optimizer = get_opt(args.opt.lower(),lr)

  if(args.dataset.lower()=='svt'):
    mydatalog = SVT(__DEF_SVT_DIR,out_size=[args.datay,args.datax])
    gtformat = 'xywh'
  elif(args.dataset.lower()=='ttt'):
    mydatalog = TTText(__DEF_TTT_DIR,out_size=None,nor=True,max_size=1280)
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

  elif(args.net=='unet'):
    # Mask Unet
    tkname = "Unet_with_{}".format(args.opt)
    trainer = UnetTrainer(isdebug=isdebug,task_name=tkname)
    mydatalog = TTText(__DEF_TTT_DIR,out_size=[args.datay,args.datax])
    model = Unet()
    los_mod = 'nor'
    part_mask_loss = True
    loss = UnetLoss(los_mod = los_mod,part_mask_loss=part_mask_loss)
    summarize += "\t Loss model: {}\n".format(los_mod)

  else:
    # label RCNN
    dif_nor = False
    tkname = "{}_with_{}_{}".format(args.name,args.dataset,args.opt) if(args.name!=None) else "LRCNN_with_{}_{}".format(args.dataset,args.opt)
    trainer = LRCNNTrainer(isdebug=isdebug,task_name=tkname,gtformat=gtformat,gen_box_by_gt=False,)
    model = Label_RCNN(num_classes=2,dif_nor=dif_nor)
    loss = LRCNNLoss(imge_size=[args.datay,args.datax],gtformat=gtformat,dif_nor=dif_nor)
  
  trainer.log_txt(summarize)
  if(not(isdebug) and args.load):
    last_model = trainer.load(model,tkname)
    if(last_model!=None):
      model = last_model
      mydatalog.setconter(trainer.data_count)

  # model.compile(
  #   optimizer=optimizer,
  #   loss=loss,
  #   )

  if(isdebug):
    for i in range(3):
      x_train, y_train = mydatalog.read_train_batch(3)
      x_val, y_val = mydatalog.read_test_batch(2)
      ret = trainer.fit(x_train,y_train,model,loss,optimizer)
      if(ret==-1):
        break
      trainer.evaluate(x_val,y_val)

  else:
    islog=False
    trainer.set_trainer(model=model,loss=loss,opt=optimizer)
    # x_train, y_train = mydatalog.read_train_batch(1)
    for i in trange(args.step):
      x_train, y_train = mydatalog.read_train_batch(args.batch)
        
      ret = trainer.fit(x_train,y_train)
      if(ret==-1):break
      if(i%args.logstp==0):
        x_val, y_val = mydatalog.read_test_batch(2)
        trainer.evaluate(x_val,y_val)
        trainer.evaluate(x_train, y_train)

      if(i and 'LR_DEC_STP' in tcfg and tcfg['LR_DEC_STP']!=0 and i%tcfg['LR_DEC_STP']==0):
        lr*=(1.0-tcfg['LR_DEC_RT'])
        trainer.set_trainer(opt=get_opt(args.opt.lower(),lr))
        trainer.log_txt("\n=======\nChange optimizer in step{}: {}, lr={}.\n=======\n".format((i+1)*args.batch,args.opt.lower(),lr))

  if(ret==0 and args.save):
    trainer.set_trainer(data_count=mydatalog.train_conter)
    trainer.save()
  time_usage = datetime.now()
  print("End at: {}.\n".format(time_usage.strftime("%Y%m%d-%H%M%S")))
  time_usage = time_usage - time_start
  print("Time usage: {} Day {} Second.\n".format(time_usage.days,time_usage.seconds))

