import os, sys
import tensorflow as tf
import numpy as np
import argparse
# from mydataset.ctw import CTW
from lib.dataloader.svt import SVT
from lib.model.config import cfg
from lib.model.label_rcnn import Label_RCNN, LRCNNLoss
from lib.frcnn_trainer import FRCNNTrainer
from lib.tflib.bbox_transform import *
from lib.tflib.evaluate_tools import *
from lib.tflib.log_tools import auto_image,save_image
from lib.dataloader.total import TTText
# 

__DEF_LOCAL_DIR = os.path.split(__file__)[0]
__DEF_DATA_DIR = os.path.join(__DEF_LOCAL_DIR,'mydataset')
__DEF_CTW_DIR = os.path.join(__DEF_DATA_DIR,'ctw')
__DEF_SVT_DIR = os.path.join(__DEF_DATA_DIR,'svt')
__DEF_TTT_DIR = os.path.join(__DEF_DATA_DIR,'totaltext')

# 
def train_lite():
  mydatalog = TTText(__DEF_TTT_DIR)
  # x_train, y_train = mydatalog.read_train_batch(1)
  loss = LRCNNLoss((360,640),gtformat='xywh')
  x_train, y_train = mydatalog.read_train_batch(10)
  x_train, y_train = mydatalog.read_train_batch(1)
  x_train, y_train = mydatalog.read_test_batch(1)
  x_train, y_train = mydatalog.read_test_batch(10)
  model = Label_RCNN()
  model(tf.zeros((1,360,640,3)))
  opt = tf.keras.optimizers.Adam(learning_rate=0.001)
  for i in range(20):
    x_train, y_train = mydatalog.read_train_batch(10)
    x_train = tf.split(x_train,x_train.shape[0],axis=0)
    for step in range(len(x_train)):
      with tf.GradientTape(persistent=False) as tape:
        tape.watch(model.trainable_variables)
        pred = model(x_train[step]/256.0)
        loss = loss(y_train[step],pred)
      grads = tape.gradient(loss, model.trainable_variables)
      opt.apply_gradients(zip(grads, model.trainable_variables))

def plt_demo():
  y1 = lambda x: np.tanh(x)
  y2 = lambda x: x+np.log(2+np.sqrt(3))
  # y2 = lambda x: x+np.tanh(x)-np.log(2+np.sqrt(3))
  y3 = lambda x: np.where(x>1,x-0.5,0.5*(x**2))
  plt.figure(num=1,figsize=[5,5])
  xarry=np.linspace(0.0,5.0,50)
  plt.plot(xarry,y1(xarry),color='blue',linewidth=1.0)
  plt.plot(xarry,y2(xarry),color='red',linewidth=1.0)
  plt.plot(xarry,y3(xarry),color='green',linewidth=1.0,linestyle='-.')
  plt.show()

if __name__ == "__main__":
  mydatalog = TTText(__DEF_TTT_DIR,out_size=[360,640])
  x_train, y_train = mydatalog.read_train_batch(1)
  model = Label_RCNN()
  model(tf.zeros((1,360,640,3)))
  pred = model(x_train[0])
  ret = pre_box_loss_by_msk(gt_mask=y_train[0],det_map=pred["l1_bbox_det"],score_map=pred["l1_score"],org_size=[360,640],use_pixel=False)
  print('end\n')