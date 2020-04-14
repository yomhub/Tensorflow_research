import os, sys
import tensorflow as tf
import numpy as np
import argparse
# from mydataset.ctw import CTW
from lib.dataloader.svt import SVT
from lib.model.config import cfg
from lib.model.label_rcnn import Label_RCNN, LRCNNLoss
from lib.model.unet import Unet, UnetLoss
from lib.frcnn_trainer import FRCNNTrainer
from lib.unet_trainer import UnetTrainer
from lib.tflib.bbox_transform import *
from lib.tflib.evaluate_tools import *
from lib.tflib.log_tools import auto_image,save_image,rf_helper
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

def rf_print():
  layers = [[3,1],[3,1],[2,2],
    [3,1],[3,1],[2,2],
    [3,1],[3,1],[3,1],[2,2],
    [3,1],[3,1],[3,1],[2,2],
    [3,1],[3,1],[3,1],[2,2],
    [3,1],
    ]
  rf_st,cod_table = rf_helper(layers,720)
  for i in range(1,len(cod_table)):
    print("Tab in {}: RF = {}, ST = {}, kernel={}, stride={}, Outsize = {}".format(
      i,rf_st[i][0],rf_st[i][1],layers[i-1][0],layers[i-1][1],len(cod_table[i])))
    for j in range(min(10,len(cod_table[i]))):
      sys.stdout.write(str(cod_table[i][j])+"||")
    if(len(cod_table[i])>10):
      sys.stdout.write('\n')
      for j in range((len(cod_table[i])-10)if(len(cod_table[i])>=20)else 10,len(cod_table[i])):
        sys.stdout.write(str(cod_table[i][j])+"||")
    sys.stdout.write('\n')

def layer_cod_gen():
  ret = feat_layer_cod_gen(
    tf.convert_to_tensor([float(44+2*8),float(44+2*8),float(8),float(8)]),
    tf.convert_to_tensor([100,100],dtype=tf.int64),
    class_num=4,
  )
  return ret

def rf_test():
  # filt = np.ones((3,3,1,1),dtype=np.float)
  filt = np.array([[0.1,0.1,0.1],[0.1,0.2,0.1],[0.1,0.1,0.1]]).reshape([3,3,1,1])
  filt1 = np.ones((1,1,1,1),dtype=np.float)
  tmp = np.zeros((1,640,640,1),dtype=np.float)
  # tmp = np.zeros((1,360,640,1),dtype=np.float)
  # tmp = np.zeros((1,720,1280,1),dtype=np.float)
  # tmp1 = np.arange(1,641,dtype=np.float).reshape([1,640])
  # tmp2 = np.arange(0,360,dtype=np.float).reshape([360,1])
  # tmp = (tmp2*640)+tmp1
  # tmp = tmp.reshape((1,360,640,1))
  tmp[0,0:153,0:153,0] = 1.0
  tmp_sp = tf.image.extract_patches(
    images=tmp,
    sizes=[1,276,276,1],
    strides=[1,32,32,1],
    rates=[1,1,1,1],
    padding='SAME',
  )
  tmp_sp = tf.reshape(tmp_sp,tmp_sp.shape[-3:-1]+[276,276]).numpy().astype(np.int)
  # for i in range(tmp_sp.shape[0]):
  #   np.savetxt('y/sp_y{}_x0.out'.format(i),tmp_sp[i,0],fmt='%d')
  # for i in range(tmp_sp.shape[1]):
  #   np.savetxt('x/sp_y0_x{}.out'.format(i),tmp_sp[0,i],fmt='%d')
  tmp_sp_b = tf.reduce_sum(tmp_sp,axis=-1)
  # B1
  tmp = tf.nn.conv2d(tmp,filters=filt,strides=[1,1,1,1],padding="SAME") # 3 1
  tmp = tf.nn.conv2d(tmp,filters=filt,strides=[1,1,1,1],padding="SAME") # 5 1
  tmp = tf.nn.max_pool2d(tmp,2,2,'VALID') # 6 2
  # B2
  tmp = tf.nn.conv2d(tmp,filters=filt,strides=[1,1,1,1],padding="SAME") # 10 2
  tmp = tf.nn.conv2d(tmp,filters=filt,strides=[1,1,1,1],padding="SAME") # 14 2
  tmp = tf.nn.max_pool2d(tmp,2,2,'VALID') # 16 4
    # B3
  tmp = tf.nn.conv2d(tmp,filters=filt,strides=[1,1,1,1],padding="SAME") # 24 4
  tmp = tf.nn.conv2d(tmp,filters=filt,strides=[1,1,1,1],padding="SAME") # 32 4
  tmp = tf.nn.conv2d(tmp,filters=filt,strides=[1,1,1,1],padding="SAME") # 40 4
  tmp = tf.nn.max_pool2d(tmp,2,2,'VALID') # 44 8
    # B4
  tmp = tf.nn.conv2d(tmp,filters=filt,strides=[1,1,1,1],padding="SAME") # 60 8
  tmp = tf.nn.conv2d(tmp,filters=filt,strides=[1,1,1,1],padding="SAME") # 76 8
  tmp = tf.nn.conv2d(tmp,filters=filt,strides=[1,1,1,1],padding="SAME") # 92 8
  tmp = tf.nn.max_pool2d(tmp,2,2,'VALID') # 100 16
    # B5
  tmp = tf.nn.conv2d(tmp,filters=filt,strides=[1,1,1,1],padding="SAME") # 132 16
  tmp = tf.nn.conv2d(tmp,filters=filt,strides=[1,1,1,1],padding="SAME") # 164 16
  tmp = tf.nn.conv2d(tmp,filters=filt,strides=[1,1,1,1],padding="SAME") # 196 16
  tmp = tf.nn.max_pool2d(tmp,2,2,'VALID') # 212 32
  # N
  tmp = tf.nn.conv2d(tmp,filters=filt,strides=[1,1,1,1],padding="SAME")
  tmp = tf.nn.conv2d(tmp,filters=filt1,strides=[1,1,1,1],padding="SAME")
  # 276,276,32,32
  return tmp
  
def t_pre_box_loss_by_msk():
  # mydatalog = TTText(__DEF_TTT_DIR,out_size=[360,640])
  # x_train, y_train = mydatalog.read_train_batch(10)
  # model = Label_RCNN()
  # model(tf.zeros((1,360,640,3)))
  # pred = model(x_train[0])
  '/am/home/yomcoding/TensorFlow/FasterRCNN/save_model/LRCNN_with_ttt_adam/20200325-203940/model'

  # for i in range(len(y_train)):
  #   ret = pre_box_loss_by_msk(gt_mask=y_train[i],det_map=pred["l1_bbox_det"],score_map=pred["l1_score"],recf_size=pred["l1_rf_s"],det_map_fom='pix',use_pixel=False)
  #   ret = pre_box_loss_by_msk(gt_mask=y_train[i],det_map=pred["l2_bbox_det"],score_map=pred["l2_score"],recf_size=pred["l2_rf_s"],det_map_fom='pix',use_pixel=False)
  #   ret = pre_box_loss_by_msk(gt_mask=y_train[i],det_map=pred["l3_bbox_det"],score_map=pred["l3_score"],recf_size=pred["l3_rf_s"],det_map_fom='pix',use_pixel=False)

if __name__ == "__main__":
  model = Unet()
  tt=tf.zeros((1,1080,1920,3))
  model(tt)
  # model.fit(tt,tf.zeros((1,1080,1920,1)))
  trainer = UnetTrainer('text',True)
  trainer.set_trainer(model=model,loss=UnetLoss(),opt=tf.keras.optimizers.Adam())
  trainer.fit()
  print('end')
